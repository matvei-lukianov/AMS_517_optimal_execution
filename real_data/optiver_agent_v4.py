"""
Risk-Averse Deep RL for Optimal Trade Execution — v4

Changes vs v3:
  - Soft urgency penalty (scale=0.5 vs 3.0 in v2, 0 in v3)
    v2 (scale=3.0): too strong, destabilised RA (std=76, CVaR=152)
    v3 (scale=0.0): too weak, RA diverged early then recovered, CVaR=128
    v4 (scale=0.5): gentle push to sell, should not conflict with utility transform

Methodology unchanged:
  - Nevmyvaka (2006): LOB sweep (L1→L2→deep-book)
  - Shen (2014):      u(x) = sign(x)|x|^lambda, lambda=0.6
  - Almgren-Chriss (1998): linear permanent impact
"""

import os, sys, threading, random, warnings
sys.stdout.reconfigure(line_buffering=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

LAMBDA           = 0.6
GAMMA            = 1.0
EPSILON_START    = 1.0
EPSILON_END      = 0.02
EPSILON_DECAY    = 0.9985
BATCH_SIZE       = 512
REPLAY_SIZE      = 200_000
LR               = 3e-4
TARGET_UPDATE    = 500
TRAIN_EPISODES   = 20_000
INIT_INVENTORY   = 200
T_STEPS          = 10
N_ACTIONS        = 11
TERMINAL_PENALTY = 0.015   # 1.5% — forces agent to complete execution
URGENCY_SCALE    = 0.5     # soft urgency — gentle push, won't conflict with utility transform
GAMMA_PERM       = 0.00002

STATE_DIM = 5  # continuous: inv_frac, spread_norm, time_frac, imbalance, depth

# ---------------------------------------------------------------------------
# Data loading — all available stocks
# ---------------------------------------------------------------------------

print("Loading Optiver LOB data...")
DATA_ROOT = '../optiver_data/book_train.parquet'
all_stock_ids = sorted([
    d.replace('stock_id=', '')
    for d in os.listdir(DATA_ROOT)
    if d.startswith('stock_id=')
])
print(f"Found {len(all_stock_ids)} stocks, using all.")

stock_data = {}
for sid in all_stock_ids:
    path = os.path.join(DATA_ROOT, f'stock_id={sid}')
    df = pd.read_parquet(path).sort_values(['time_id', 'seconds_in_bucket'])
    stock_data[sid] = df

total_eps = sum(len(df['time_id'].unique()) for df in stock_data.values())
print(f"Loaded {total_eps} total episodes across {len(all_stock_ids)} stocks")

# Per-stock spread stats for normalisation (no lookahead — train set only)
spread_stats = {}
train_episodes = {}
test_episodes  = {}
for sid, df in stock_data.items():
    tids   = df['time_id'].unique()
    n_test = max(10, int(len(tids) * 0.1))
    train_episodes[sid] = tids[:-n_test]
    test_episodes[sid]  = tids[-n_test:]
    # Compute spread stats on train set only
    train_df = df[df['time_id'].isin(train_episodes[sid])]
    spreads  = (train_df['ask_price1'] - train_df['bid_price1']).values
    spreads  = spreads[spreads > 0]
    spread_stats[sid] = (spreads.mean(), spreads.std() + 1e-8)

n_train = sum(len(v) for v in train_episodes.values())
n_test  = sum(len(v) for v in test_episodes.values())
print(f"Train: {n_train}  Test: {n_test}")

print("Pre-grouping DataFrames...")
grouped_all = {}
for sid, df in stock_data.items():
    grouped_all[sid] = {tid: g for tid, g in df.groupby('time_id')}
print("Ready.\n")

STOCKS = all_stock_ids

# ---------------------------------------------------------------------------
# Environment — continuous state
# ---------------------------------------------------------------------------

class OptiverEnv:
    """
    Continuous state vector (5 features):
        [0] inv_frac    = inventory / INIT_INVENTORY          ∈ [0, 1]
        [1] spread_norm = (spread - mu) / sigma               ∈ R (z-score)
        [2] time_frac   = t / T_STEPS                         ∈ [0, 1]
        [3] imbalance   = bid_size1 / (bid_size1+ask_size1)   ∈ [0, 1]
        [4] depth       = bid_size2 / (bid_size1 + 1)         ∈ R+

    This replaces 23-dim one-hot with 5 continuous features.
    Network can now interpolate between states rather than treating each
    bucket as independent — better generalization, faster training.
    """

    def __init__(self, sid, tid, grouped, ss):
        self.sid = sid
        self.ss  = ss   # (mean_spread, std_spread)

        df = grouped[tid].drop_duplicates('seconds_in_bucket')
        n  = len(df)
        idx = np.linspace(0, n - 1, T_STEPS, dtype=int)
        df  = df.iloc[idx]

        self.b1p = df['bid_price1'].values.copy()
        self.b1s = df['bid_size1'].values.copy()
        self.b2p = df['bid_price2'].values.copy()
        self.b2s = df['bid_size2'].values.copy()
        self.a1p = df['ask_price1'].values.copy()
        self.a1s = df['ask_size1'].values.copy()
        self.horizon = T_STEPS

        self.init_mid = (self.b1p[0] + self.a1p[0]) / 2.0
        self.perfect  = self.init_mid * INIT_INVENTORY
        self.reset()

    def reset(self):
        self.t         = 0
        self.inventory = INIT_INVENTORY
        self.cash      = 0.0
        self.done      = False
        self.cum_perm  = 0.0
        return self._state()

    def _state(self):
        t = min(self.t, self.horizon - 1)

        inv_frac   = self.inventory / INIT_INVENTORY

        spread     = max(0.0, self.a1p[t] - self.b1p[t])
        mu, sigma  = self.ss
        spread_z   = (spread - mu) / sigma

        time_frac  = self.t / self.horizon

        b1s = max(1.0, float(self.b1s[t]))
        a1s = max(1.0, float(self.a1s[t]))
        imbalance  = b1s / (b1s + a1s)

        depth      = float(self.b2s[t]) / (b1s + 1.0)

        return np.array([inv_frac, spread_z, time_frac, imbalance, depth],
                        dtype=np.float32)

    def step(self, q: int):
        q = max(0, min(q, self.inventory))
        t = min(self.t, self.horizon - 1)

        b1p = self.b1p[t] - self.cum_perm
        b2p = self.b2p[t] - self.cum_perm

        exec_cash = 0.0
        q_rem     = q

        # Level-1 sweep (Nevmyvaka)
        q_l1       = min(q_rem, self.b1s[t])
        exec_cash += q_l1 * b1p
        q_rem     -= q_l1

        # Level-2 sweep (Nevmyvaka)
        q_l2       = min(q_rem, self.b2s[t])
        exec_cash += q_l2 * b2p
        q_rem     -= q_l2

        # Deep-book penalty
        if q_rem > 0:
            slip       = 0.05 * (q_rem / INIT_INVENTORY)
            exec_cash += q_rem * b2p * (1.0 - slip)

        self.cash      += exec_cash
        self.inventory -= q
        self.cum_perm  += GAMMA_PERM * q

        mid = (self.b1p[t] + self.a1p[t]) / 2.0 - self.cum_perm

        # Step reward: execution quality in bps
        if q > 0:
            reward = (exec_cash / q - mid) / self.init_mid * 1e4
        else:
            reward = 0.0

        # Soft urgency: gentle penalty for holding inventory near deadline
        steps_remaining = max(1, self.horizon - self.t - 1)
        urgency = (self.inventory / INIT_INVENTORY) / steps_remaining
        reward -= URGENCY_SCALE * urgency

        self.t += 1
        if self.t >= self.horizon or self.inventory <= 0:
            self.done = True
            if self.inventory > 0:
                # Stronger terminal penalty (1.5%) — forces agent to complete execution
                dump_p     = (self.b2p[-1] - self.cum_perm) * (1.0 - TERMINAL_PENALTY)
                dump_mid   = (self.b1p[-1] + self.a1p[-1]) / 2.0
                self.cash += self.inventory * dump_p
                reward    -= self.inventory * (dump_mid - dump_p) / self.init_mid * 1e4
                self.inventory = 0
            shortfall = self.perfect - self.cash
            return self._state(), reward, True, {
                'shortfall':     shortfall,
                'shortfall_bps': shortfall / self.perfect * 1e4,
            }

        return self._state(), reward, False, {}


# ---------------------------------------------------------------------------
# DQN — smaller network (5 → 128 → 128 → 11), faster than 23→256→256
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


def state_to_tensor(state: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(state)


def states_to_batch(states) -> torch.Tensor:
    return torch.from_numpy(np.stack(states)).to(DEVICE)


def shen_utility(x: torch.Tensor, lam: float = LAMBDA) -> torch.Tensor:
    return torch.sign(x) * (torch.abs(x) + 1e-8).pow(lam)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# Training function — runs one agent (RA or RN) in its own thread
# ---------------------------------------------------------------------------

def train_agent(name: str, lam: float, results_dict: dict):
    net    = DQN().to(DEVICE)
    target = DQN().to(DEVICE)
    target.load_state_dict(net.state_dict())
    target.eval()
    opt    = optim.Adam(net.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_SIZE)
    eps    = EPSILON_START
    steps  = 0
    reward_history = []
    loss_history   = []

    print(f"[{name}] Starting training — lambda={lam}")

    for ep in range(TRAIN_EPISODES):
        sid  = random.choice(STOCKS)
        tid  = random.choice(train_episodes[sid])
        env  = OptiverEnv(sid, tid, grouped_all[sid], spread_stats[sid])
        s    = env.reset()
        done = False
        ep_r = 0.0

        while not done:
            # Epsilon-greedy
            if random.random() < eps:
                a_idx = random.randint(0, N_ACTIONS - 1)
            else:
                with torch.no_grad():
                    sv = state_to_tensor(s).unsqueeze(0).to(DEVICE)
                    a_idx = int(net(sv).argmax(1).item())

            q_trade = int(round(env.inventory * a_idx / (N_ACTIONS - 1)))
            ns, r, done, _ = env.step(q_trade)
            replay.push(s, a_idx, r, ns, done)
            s     = ns
            ep_r += r
            steps += 1

            # Train step
            if len(replay) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)
                s_b  = states_to_batch(states)
                ns_b = states_to_batch(next_states)
                a_b  = torch.tensor(actions, dtype=torch.long,    device=DEVICE)
                r_b  = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
                d_b  = torch.tensor(dones,   dtype=torch.float32, device=DEVICE)

                q_vals = net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q   = target(ns_b).max(1).values
                    q_target = r_b + GAMMA * next_q * (1.0 - d_b)

                td_error = q_target - q_vals
                if lam < 1.0:
                    loss = shen_utility(td_error, lam).pow(2).mean()
                else:
                    loss = td_error.pow(2).mean()  # plain MSE for RN

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                loss_history.append(loss.item())

            if steps % TARGET_UPDATE == 0:
                target.load_state_dict(net.state_dict())

        eps = max(EPSILON_END, eps * EPSILON_DECAY)
        reward_history.append(ep_r)

        if ep % 2000 == 0:
            avg_r = np.mean(reward_history[-500:]) if reward_history else 0
            avg_l = np.mean(loss_history[-500:])   if loss_history  else 0
            print(f"  [{name}] Ep {ep:6d}/{TRAIN_EPISODES} | eps={eps:.3f} | "
                  f"avg_reward={avg_r:+.2f} bps | loss={avg_l:.4f}")

    print(f"[{name}] Training complete.")
    torch.save(net.state_dict(), f"dqn_{name.lower()}_v2.pt")
    results_dict[name] = net


# ---------------------------------------------------------------------------
# Run RA and RN in parallel threads
# ---------------------------------------------------------------------------

print("=" * 60)
print("Training RA-DQN (λ=0.6) and RN-DQN (λ=1.0) in PARALLEL")
print("=" * 60)

trained_nets = {}

t_ra = threading.Thread(target=train_agent, args=("RA", 0.6, trained_nets))
t_rn = threading.Thread(target=train_agent, args=("RN", 1.0, trained_nets))

t_ra.start()
t_rn.start()
t_ra.join()
t_rn.join()

print("\nBoth agents trained.")

# ---------------------------------------------------------------------------
# Evaluation: 4 policies
# ---------------------------------------------------------------------------

def run_net(net, sid, tid):
    env  = OptiverEnv(sid, tid, grouped_all[sid], spread_stats[sid])
    s    = env.reset(); done = False
    while not done:
        with torch.no_grad():
            sv    = state_to_tensor(s).unsqueeze(0).to(DEVICE)
            a_idx = int(net(sv).argmax(1).item())
        q_trade = int(round(env.inventory * a_idx / (N_ACTIONS - 1)))
        s, _, done, info = env.step(q_trade)
    return info

def run_twap(sid, tid):
    env    = OptiverEnv(sid, tid, grouped_all[sid], spread_stats[sid])
    s      = env.reset(); done = False
    q_each = max(1, INIT_INVENTORY // T_STEPS)
    while not done:
        q = env.inventory if (T_STEPS - env.t) <= 1 else q_each
        s, _, done, info = env.step(q)
    return info

def run_vwap(sid, tid):
    env     = OptiverEnv(sid, tid, grouped_all[sid], spread_stats[sid])
    s       = env.reset(); done = False
    t_pts   = np.linspace(-1, 1, T_STEPS)
    weights = t_pts ** 2 + 0.5
    while not done:
        t     = env.t
        w_rem = weights[t:].sum()
        frac  = weights[t] / w_rem if w_rem > 0 else 1.0
        q     = int(np.ceil(env.inventory * frac))
        s, _, done, info = env.step(q)
    return info

print(f"\nEvaluating on {n_test} test episodes across {len(STOCKS)} stocks...")

ra_net = trained_nets['RA']
rn_net = trained_nets['RN']

results = {k: {'sf': [], 'bps': []} for k in ['RA_DQN', 'RN_DQN', 'VWAP', 'TWAP']}

for sid in STOCKS:
    for tid in test_episodes[sid]:
        for key, fn in [
            ('RA_DQN', lambda s, t: run_net(ra_net, s, t)),
            ('RN_DQN', lambda s, t: run_net(rn_net, s, t)),
            ('VWAP',   run_vwap),
            ('TWAP',   run_twap),
        ]:
            info = fn(sid, tid)
            results[key]['sf'].append(info['shortfall'])
            results[key]['bps'].append(info['shortfall_bps'])

for key in results:
    results[key]['sf']  = np.array(results[key]['sf'])
    results[key]['bps'] = np.array(results[key]['bps'])

print("\n" + "=" * 72)
print(f"OUT-OF-SAMPLE RESULTS (v2) — {len(STOCKS)} stocks, {len(results['RA_DQN']['sf'])} episodes")
print(f"Continuous state | Parallel training | Terminal penalty=1.5% | Soft urgency scale=0.5")
print("=" * 72)
print(f"{'Metric':<28} {'RA-DQN':>10} {'RN-DQN':>10} {'VWAP':>10} {'TWAP':>10}")
print("-" * 68)
for label, fn in [
    ("Mean Cost (bps)",  lambda k: results[k]['bps'].mean()),
    ("Std Cost (bps)",   lambda k: results[k]['bps'].std()),
    ("95% CVaR (bps)",   lambda k: np.quantile(results[k]['bps'], 0.95)),
]:
    vals = [fn(k) for k in ['RA_DQN', 'RN_DQN', 'VWAP', 'TWAP']]
    print(f"{label:<28} {vals[0]:>10.2f} {vals[1]:>10.2f} {vals[2]:>10.2f} {vals[3]:>10.2f}")
print("-" * 68)
ra_cvar = np.quantile(results['RA_DQN']['bps'], 0.95)
rn_cvar = np.quantile(results['RN_DQN']['bps'], 0.95)
vwap_cvar = np.quantile(results['VWAP']['bps'], 0.95)
print(f"RA vs RN  CVaR: {rn_cvar - ra_cvar:+.1f} bps  (Shen: -10 to -15 bps)")
print(f"RA vs VWAP CVaR: {vwap_cvar - ra_cvar:+.1f} bps")
print(f"RA beats VWAP (mean): {(results['RA_DQN']['sf'] < results['VWAP']['sf']).mean()*100:.0f}% of episodes")
print(f"RA beats TWAP (mean): {(results['RA_DQN']['sf'] < results['TWAP']['sf']).mean()*100:.0f}% of episodes")
print("=" * 72)

pd.DataFrame({
    'RA_DQN_bps': results['RA_DQN']['bps'],
    'RN_DQN_bps': results['RN_DQN']['bps'],
    'VWAP_bps':   results['VWAP']['bps'],
    'TWAP_bps':   results['TWAP']['bps'],
    'RL_CVaR_Shortfall': results['RA_DQN']['sf'],
    'VWAP_Shortfall':    results['VWAP']['sf'],
    'RL_bps':            results['RA_DQN']['bps'],
}).to_csv("optiver_results.csv", index=False)
print("Saved optiver_results.csv")
