"""
Risk-Averse Deep RL for Optimal Trade Execution — v3

Changes vs v1:
  - Continuous state (5 features) instead of one-hot (23 dim)
      inventory/Q, spread_norm, t/T, LOB_imbalance, book_depth
  - RA-DQN and RN-DQN trained in parallel threads (both on GPU simultaneously)
  - Stronger terminal penalty (1.5%), NO urgency penalty
  - All 112 stocks (not just 20) for better generalization

Changes vs v2:
  - Removed urgency penalty: it conflicted with Shen utility transform,
    both rescale gradients → RA training instability (std=76 vs 30 for RN)
  - Terminal penalty 1.5% alone forces completion without destabilising RA

Methodology unchanged:
  - Nevmyvaka (2006): LOB sweep (L1→L2→deep-book)
  - Shen (2014):      u(x) = sign(x)|x|^lambda on TD-error (λ default 0.6, override SHEN_LAMBDA)
  - Almgren-Chriss (1998): linear permanent impact
"""

import os, sys, threading, random, warnings, json, shutil
from datetime import datetime
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

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = f"optiver_results_v3_{RUN_ID}.csv"
MANIFEST_PATH = f"optiver_v3_manifest_{RUN_ID}.json"
print(f"Run id (tagged outputs, no overwrites): {RUN_ID}\n")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# RA-only (Shen Eq.9–10). Override for λ-sweeps: SHEN_LAMBDA=0.55 python3.9 ...
RA_LAMBDA        = float(os.environ.get("SHEN_LAMBDA", "0.6"))
LAMBDA           = RA_LAMBDA   # default for shen_utility()
GAMMA            = 1.0
EPSILON_START    = 1.0
EPSILON_END      = 0.02
EPSILON_DECAY    = 0.9985
BATCH_SIZE       = 512
REPLAY_SIZE      = 200_000
LR               = 3e-4
TARGET_UPDATE    = 500
TRAIN_EPISODES   = int(os.environ.get("OPTIVER_V3_TRAIN_EPISODES", "20000"))
# Hold out tail of *train* time_ids for validation metrics (no test leakage).
VAL_FRAC         = float(os.environ.get("OPTIVER_V3_VAL_FRAC", "0"))
# Cap val episodes for metrics (full val can be 40k+ → looks “hung”). 0 = no cap.
VAL_EVAL_MAX     = int(os.environ.get("OPTIVER_V3_VAL_MAX_EPISODES", "8000"))
# If 1: after training + optional val metrics, skip full OOS eval / CSV / plots.
SWEEP_MODE       = os.environ.get("OPTIVER_V3_SWEEP_MODE", "0") == "1"
# If 1: train RA only (faster). RN for val/full eval loaded from OPTIVER_V3_RN_WEIGHTS if set.
RA_ONLY          = os.environ.get("OPTIVER_V3_RA_ONLY", "0") == "1"
RN_WEIGHTS_PATH  = os.environ.get("OPTIVER_V3_RN_WEIGHTS", "dqn_rn_v3_latest.pt")
INIT_INVENTORY   = 200
# Monte Carlo sample size for trajectory plots (mean ± std band)
TRACE_PLOT_EPISODES = 400
T_STEPS          = 10
N_ACTIONS        = 11
TERMINAL_PENALTY = 0.015   # 1.5% — forces agent to complete execution
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

train_sample_pool = train_episodes
val_episodes: dict = {}
if VAL_FRAC > 0:
    train_sample_pool = {}
    val_episodes = {}
    for sid in STOCKS:
        tr = list(train_episodes[sid])
        n_tr = len(tr)
        k = max(1, int(n_tr * VAL_FRAC))
        if k >= n_tr:
            k = max(1, n_tr // 5)
        train_sample_pool[sid] = tr[:-k]
        val_episodes[sid] = tr[-k:]
    n_fit = sum(len(v) for v in train_sample_pool.values())
    n_val = sum(len(v) for v in val_episodes.values())
    print(f"VAL_FRAC={VAL_FRAC}: train-fit episodes={n_fit}, val episodes={n_val}\n")

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

        step_exec_price = float(exec_cash / q) if q > 0 else float(mid)

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
                'shortfall':       shortfall,
                'shortfall_bps':   shortfall / self.perfect * 1e4,
                'step_exec_price': step_exec_price,
                'inv_after':       int(self.inventory),
            }

        return self._state(), reward, False, {
            'step_exec_price': step_exec_price,
            'inv_after':       int(self.inventory),
        }


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

def train_agent(name: str, lam: float, results_dict: dict, weight_paths_dict: dict):
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
        tid  = random.choice(train_sample_pool[sid])
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
    wpath = f"dqn_{name.lower()}_v3_{RUN_ID}.pt"
    torch.save(net.state_dict(), wpath)
    weight_paths_dict[name] = wpath
    print(f"  [{name}] Saved {wpath}")
    results_dict[name] = net


# ---------------------------------------------------------------------------
# Run RA and RN in parallel threads
# ---------------------------------------------------------------------------

print("=" * 60)
if RA_ONLY:
    print(f"Training RA-DQN only (λ={RA_LAMBDA})  [OPTIVER_V3_RA_ONLY=1]")
else:
    print(f"Training RA-DQN (λ={RA_LAMBDA}) and RN-DQN (λ=1.0) in PARALLEL")
print(f"TRAIN_EPISODES={TRAIN_EPISODES}  VAL_FRAC={VAL_FRAC}  SWEEP_MODE={SWEEP_MODE}  RA_ONLY={RA_ONLY}")
print("=" * 60)

trained_nets = {}
weight_paths = {}

if RA_ONLY:
    train_agent("RA", RA_LAMBDA, trained_nets, weight_paths)
else:
    t_ra = threading.Thread(target=train_agent, args=("RA", RA_LAMBDA, trained_nets, weight_paths))
    t_rn = threading.Thread(target=train_agent, args=("RN", 1.0, trained_nets, weight_paths))
    t_ra.start()
    t_rn.start()
    t_ra.join()
    t_rn.join()

print("\nTraining phase done." if RA_ONLY else "\nBoth agents trained.")
for nm, pth in weight_paths.items():
    if pth and os.path.isfile(pth):
        latest = f"dqn_{nm.lower()}_v3_latest.pt"
        shutil.copy2(pth, latest)
        print(f"Copied weights -> {latest}")

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


def _pad_inv(inv_trace, T):
    inv_trace = list(inv_trace)
    while len(inv_trace) < T + 1:
        inv_trace.append(0.0)
    return np.array(inv_trace[: T + 1], dtype=np.float64)


def _pad_prices(prices, T):
    prices = list(prices)
    if len(prices) < T:
        last = prices[-1] if prices else float("nan")
        pad = last if np.isfinite(last) else float("nan")
        while len(prices) < T:
            prices.append(pad)
    return np.array(prices[:T], dtype=np.float64)


def trace_dqn(net, sid, tid):
    env = OptiverEnv(sid, tid, grouped_all[sid], spread_stats[sid])
    s = env.reset()
    inv_w = [float(env.inventory)]
    prices = []
    done = False
    T = env.horizon
    while not done:
        with torch.no_grad():
            sv = state_to_tensor(s).unsqueeze(0).to(DEVICE)
            a_idx = int(net(sv).argmax(1).item())
        q_trade = int(round(env.inventory * a_idx / (N_ACTIONS - 1)))
        s, _, done, info = env.step(q_trade)
        prices.append(info["step_exec_price"])
        inv_w.append(float(env.inventory))
    return _pad_inv(inv_w, T), _pad_prices(prices, T)


def trace_vwap(sid, tid):
    env = OptiverEnv(sid, tid, grouped_all[sid], spread_stats[sid])
    s = env.reset()
    inv_w = [float(env.inventory)]
    prices = []
    t_pts = np.linspace(-1, 1, T_STEPS)
    weights = t_pts ** 2 + 0.5
    done = False
    T = env.horizon
    while not done:
        t = env.t
        w_rem = weights[t:].sum()
        frac = weights[t] / w_rem if w_rem > 0 else 1.0
        q = int(np.ceil(env.inventory * frac))
        s, _, done, info = env.step(q)
        prices.append(info["step_exec_price"])
        inv_w.append(float(env.inventory))
    return _pad_inv(inv_w, T), _pad_prices(prices, T)


def plot_trajectory_figure(ra_net, rn_net, n_eps: int) -> str:
    import matplotlib.pyplot as plt

    BLUE = "#2563EB"
    ORANGE = "#EA580C"
    RED = "#DC2626"

    plt.rcParams.update({"font.size": 11, "font.family": "serif"})
    all_pairs = [(sid, tid) for sid in STOCKS for tid in test_episodes[sid]]
    rng = random.Random(42)
    n_sample = min(n_eps, len(all_pairs))
    sample_pairs = rng.sample(all_pairs, n_sample)

    inv_ra, pr_ra, inv_vw, pr_vw = [], [], [], []
    inv_rn, pr_rn = [], []
    for sid, tid in sample_pairs:
        irr, prr = trace_dqn(ra_net, sid, tid)
        ivw, pvw = trace_vwap(sid, tid)
        inv_ra.append(irr)
        pr_ra.append(prr)
        inv_vw.append(ivw)
        pr_vw.append(pvw)
        if rn_net is not None:
            irn, prn = trace_dqn(rn_net, sid, tid)
            inv_rn.append(irn)
            pr_rn.append(prn)

    inv_ra = np.stack(inv_ra)
    pr_ra = np.stack(pr_ra)
    inv_vw = np.stack(inv_vw)
    pr_vw = np.stack(pr_vw)
    if rn_net is not None:
        inv_rn = np.stack(inv_rn)
        pr_rn = np.stack(pr_rn)

    tau = np.linspace(0.0, 1.0, T_STEPS + 1)
    tau_p = (np.arange(T_STEPS) + 0.5) / T_STEPS

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    m_vw, s_vw = inv_vw.mean(0), inv_vw.std(0)
    m_ra, s_ra = inv_ra.mean(0), inv_ra.std(0)
    axes[0].plot(tau, m_vw, color=BLUE, lw=2.2, label="VWAP (deterministic schedule)")
    axes[0].fill_between(tau, m_vw - s_vw, m_vw + s_vw, color=BLUE, alpha=0.12)
    axes[0].plot(tau, m_ra, color=ORANGE, lw=2.2, label="RA-DQN")
    axes[0].fill_between(tau, m_ra - s_ra, m_ra + s_ra, color="#9CA3AF", alpha=0.28)
    if rn_net is not None:
        m_rn, s_rn = inv_rn.mean(0), inv_rn.std(0)
        axes[0].plot(tau, m_rn, color=RED, lw=2.2, label="RN-DQN")
        axes[0].fill_between(tau, m_rn - s_rn, m_rn + s_rn, color="#FCA5A5", alpha=0.35)
    axes[0].set_xlabel("Normalized time")
    axes[0].set_ylabel("Shares")
    axes[0].set_title("Inventory trajectory")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(bottom=0)

    m_pv, s_pv = pr_vw.mean(0), pr_vw.std(0)
    m_pra, s_pra = pr_ra.mean(0), pr_ra.std(0)
    axes[1].plot(tau_p, m_pv, color=BLUE, lw=2.2, label="VWAP (deterministic schedule)")
    axes[1].fill_between(tau_p, m_pv - s_pv, m_pv + s_pv, color=BLUE, alpha=0.12)
    axes[1].plot(tau_p, m_pra, color=ORANGE, lw=2.2, label="RA-DQN")
    axes[1].fill_between(tau_p, m_pra - s_pra, m_pra + s_pra, color="#9CA3AF", alpha=0.28)
    if rn_net is not None:
        m_prn, s_prn = pr_rn.mean(0), pr_rn.std(0)
        axes[1].plot(tau_p, m_prn, color=RED, lw=2.2, label="RN-DQN")
        axes[1].fill_between(tau_p, m_prn - s_prn, m_prn + s_prn, color="#FCA5A5", alpha=0.35)
    axes[1].set_xlabel("Normalized time")
    axes[1].set_ylabel("Price")
    axes[1].set_title("Execution price (per step)")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_xlim(0, 1)

    sub = "RA-DQN vs RN-DQN vs VWAP" if rn_net is not None else "RA-DQN vs VWAP"
    fig.suptitle(
        f"{sub} — {n_sample} OOS episodes (mean ± std)\n"
        f"Optiver LOB | T={T_STEPS} | run {RUN_ID}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = f"trajectories_v3_{RUN_ID}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    return out


ra_net = trained_nets['RA']
rn_net = trained_nets.get("RN")

if RA_ONLY:
    if os.path.isfile(RN_WEIGHTS_PATH):
        rn_net = DQN().to(DEVICE)
        rn_net.load_state_dict(torch.load(RN_WEIGHTS_PATH, map_location=DEVICE))
        rn_net.eval()
        trained_nets["RN"] = rn_net
        print(f"Loaded RN weights from {RN_WEIGHTS_PATH} (not retrained).")
    else:
        print(
            f"WARNING: RA_ONLY=1 but {RN_WEIGHTS_PATH} not found — "
            "val RN metrics and RN_DQN in full eval will be skipped.",
            flush=True,
        )


def _flat_val_pairs(episode_dict, max_n: int):
    pairs = [(sid, tid) for sid in STOCKS for tid in episode_dict[sid]]
    n_all = len(pairs)
    if max_n > 0 and n_all > max_n:
        rng = random.Random(42)
        pairs = rng.sample(pairs, max_n)
    return pairs, n_all


def _collect_policy_bps(net, pairs):
    bps = []
    n = len(pairs)
    for i, (sid, tid) in enumerate(pairs):
        if n > 3000 and i > 0 and i % 2000 == 0:
            print(f"  val eval {i}/{n} ...", flush=True)
        info = run_net(net, sid, tid)
        bps.append(info['shortfall_bps'])
    return np.asarray(bps, dtype=np.float64)


if VAL_FRAC > 0 and val_episodes and sum(len(v) for v in val_episodes.values()) > 0:
    val_pairs, n_val_all = _flat_val_pairs(val_episodes, VAL_EVAL_MAX)
    print(f"\nVal episodes: total={n_val_all}, evaluating={len(val_pairs)} (OPTIVER_V3_VAL_MAX_EPISODES={VAL_EVAL_MAX})", flush=True)
    ra_v = _collect_policy_bps(ra_net, val_pairs)
    summary = {
        "run_id": RUN_ID,
        "ra_lambda": RA_LAMBDA,
        "train_episodes": TRAIN_EPISODES,
        "val_frac": VAL_FRAC,
        "n_val_total": int(n_val_all),
        "n_val_eval": int(len(ra_v)),
        "ra_val_mean_bps": float(ra_v.mean()),
        "ra_val_q95_bps": float(np.quantile(ra_v, 0.95)),
    }
    rn_net = trained_nets.get("RN")
    if rn_net is not None:
        rn_v = _collect_policy_bps(rn_net, val_pairs)
        summary["rn_val_mean_bps"] = float(rn_v.mean())
        summary["rn_val_q95_bps"] = float(np.quantile(rn_v, 0.95))
    with open("optiver_v3_last_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nValidation (held-out tail of train time_ids):")
    print(f"  RA  mean={summary['ra_val_mean_bps']:.2f} bps  q95={summary['ra_val_q95_bps']:.2f}")
    if rn_net is not None:
        print(f"  RN  mean={summary['rn_val_mean_bps']:.2f} bps  q95={summary['rn_val_q95_bps']:.2f}")
    print("Saved optiver_v3_last_metrics.json")

if SWEEP_MODE:
    print("\nOPTIVER_V3_SWEEP_MODE=1 — skipping full OOS test, CSV, trajectory plot.")
    raise SystemExit(0)


print(f"\nEvaluating on {n_test} test episodes across {len(STOCKS)} stocks...")

policies = [
    ('RA_DQN', lambda s, t: run_net(ra_net, s, t)),
    ('VWAP',   run_vwap),
    ('TWAP',   run_twap),
]
rn_eval = trained_nets.get("RN")
if rn_eval is not None:
    policies.insert(1, ('RN_DQN', lambda s, t: run_net(rn_eval, s, t)))

results = {k: {'sf': [], 'bps': []} for k, _ in policies}

for sid in STOCKS:
    for tid in test_episodes[sid]:
        for key, fn in policies:
            info = fn(sid, tid)
            results[key]['sf'].append(info['shortfall'])
            results[key]['bps'].append(info['shortfall_bps'])

for key in results:
    results[key]['sf']  = np.array(results[key]['sf'])
    results[key]['bps'] = np.array(results[key]['bps'])

print("\n" + "=" * 72)
print(f"OUT-OF-SAMPLE RESULTS (v2) — {len(STOCKS)} stocks, {len(results['RA_DQN']['sf'])} episodes")
print(f"Continuous state | Parallel training | Terminal penalty=1.5% | No urgency penalty")
print("=" * 72)
metric_keys = [k for k in ('RA_DQN', 'RN_DQN', 'VWAP', 'TWAP') if k in results]
hdr = f"{'Metric':<28}" + "".join(f"{k.replace('_DQN','').replace('_','-'):>12}" for k in metric_keys)
print(hdr)
print("-" * (28 + 12 * len(metric_keys)))
for label, fn in [
    ("Mean Cost (bps)",  lambda k: results[k]['bps'].mean()),
    ("Std Cost (bps)",   lambda k: results[k]['bps'].std()),
    ("95% CVaR (bps)",   lambda k: np.quantile(results[k]['bps'], 0.95)),
]:
    vals = [fn(k) for k in metric_keys]
    cells = "".join(f"{v:>12.2f}" for v in vals)
    print(f"{label:<28}{cells}")
print("-" * (28 + 12 * len(metric_keys)))
ra_cvar = np.quantile(results['RA_DQN']['bps'], 0.95)
vwap_cvar = np.quantile(results['VWAP']['bps'], 0.95)
if 'RN_DQN' in results:
    rn_cvar = np.quantile(results['RN_DQN']['bps'], 0.95)
    print(f"RA vs RN  CVaR: {rn_cvar - ra_cvar:+.1f} bps  (Shen: -10 to -15 bps)")
print(f"RA vs VWAP CVaR: {vwap_cvar - ra_cvar:+.1f} bps")
print(f"RA beats VWAP (mean): {(results['RA_DQN']['sf'] < results['VWAP']['sf']).mean()*100:.0f}% of episodes")
print(f"RA beats TWAP (mean): {(results['RA_DQN']['sf'] < results['TWAP']['sf']).mean()*100:.0f}% of episodes")
print("=" * 72)

_df = {
    'RA_DQN_bps': results['RA_DQN']['bps'],
    'VWAP_bps':   results['VWAP']['bps'],
    'TWAP_bps':   results['TWAP']['bps'],
    'RL_CVaR_Shortfall': results['RA_DQN']['sf'],
    'VWAP_Shortfall':    results['VWAP']['sf'],
    'RL_bps':            results['RA_DQN']['bps'],
}
if 'RN_DQN' in results:
    _df['RN_DQN_bps'] = results['RN_DQN']['bps']
else:
    _df['RN_DQN_bps'] = np.full(len(results['RA_DQN']['bps']), np.nan)
pd.DataFrame(_df).to_csv(CSV_PATH, index=False)
print(f"Saved {CSV_PATH}")
shutil.copy2(CSV_PATH, "optiver_results_v3_latest.csv")
print("Copied -> optiver_results_v3_latest.csv")

trajectory_png = plot_trajectory_figure(ra_net, trained_nets.get("RN"), TRACE_PLOT_EPISODES)
shutil.copy2(trajectory_png, "trajectories_v3_latest.png")
print("Copied -> trajectories_v3_latest.png")

manifest = {
    "run_id": RUN_ID,
    "csv": CSV_PATH,
    "csv_latest_copy": "optiver_results_v3_latest.csv",
    "weights": dict(weight_paths),
    "trajectory_plot": trajectory_png,
    "trajectory_plot_latest_copy": "trajectories_v3_latest.png",
    "n_stocks": len(STOCKS),
    "n_test_episodes": int(sum(len(v) for v in test_episodes.values())),
    "trace_plot_episodes": TRACE_PLOT_EPISODES,
}
with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Saved {MANIFEST_PATH}")
