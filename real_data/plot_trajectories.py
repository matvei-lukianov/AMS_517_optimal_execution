"""
Execution trajectory plot — matches original trajectories_v3 style:
  Left:  inventory trajectory (shares), mean ± std
  Right: execution price per step, mean ± std
Fixed colors: VWAP=green, RA=blue, RN=red (was orange≈red, now clearly distinct)
"""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INIT_INVENTORY   = 200
T_STEPS          = 10
N_ACTIONS        = 11
STATE_DIM        = 5
TERMINAL_PENALTY = 0.015
URGENCY_SCALE    = 0.5
GAMMA_PERM       = 0.00002

COL_VWAP = "#16A34A"   # green
COL_RA   = "#2563EB"   # blue
COL_RN   = "#DC2626"   # red

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, N_ACTIONS),
        )
    def forward(self, x):
        return self.net(x)

ra_net = DQN().to(DEVICE)
rn_net = DQN().to(DEVICE)
ra_net.load_state_dict(torch.load("dqn_ra_v3_latest.pt", map_location=DEVICE))
rn_net.load_state_dict(torch.load("dqn_rn_v3_latest.pt", map_location=DEVICE))
ra_net.eval(); rn_net.eval()

# Load stock 0 test episodes
DATA_ROOT = '../optiver_data/book_train.parquet'
sid = '0'
df = pd.read_parquet(os.path.join(DATA_ROOT, f'stock_id={sid}')).sort_values(
    ['time_id', 'seconds_in_bucket'])
grouped = {tid: g for tid, g in df.groupby('time_id')}
tids = df['time_id'].unique()
n_test = max(10, int(len(tids) * 0.1))
test_tids = tids[-n_test:]
train_tids = tids[:-n_test]
train_df = df[df['time_id'].isin(train_tids)]
spreads = (train_df['ask_price1'] - train_df['bid_price1']).values
spreads = spreads[spreads > 0]
ss = (spreads.mean(), spreads.std() + 1e-8)

print(f"Running {len(test_tids)} test episodes...")

def get_lob(tid):
    g = grouped[tid].drop_duplicates('seconds_in_bucket')
    idx = np.linspace(0, len(g)-1, T_STEPS, dtype=int)
    g = g.iloc[idx]
    return (g['bid_price1'].values.copy(),
            g['bid_size1'].values.copy(),
            g['bid_price2'].values.copy(),
            g['bid_size2'].values.copy(),
            g['ask_price1'].values.copy(),
            g['ask_size1'].values.copy())

def get_state(inventory, t, b1p, b1s, b2p, b2s, a1p, a1s, cum_perm):
    ti = min(t, T_STEPS-1)
    inv_frac  = inventory / INIT_INVENTORY
    spread_z  = (max(0.0, a1p[ti] - b1p[ti]) - ss[0]) / ss[1]
    time_frac = t / T_STEPS
    b1sv = max(1.0, float(b1s[ti]))
    a1sv = max(1.0, float(a1s[ti]))
    imbal = b1sv / (b1sv + a1sv)
    depth = float(b2s[ti]) / (b1sv + 1.0)
    return np.array([inv_frac, spread_z, time_frac, imbal, depth], dtype=np.float32)

def execute_step(q, t, b1p, b1s, b2p, b2s, a1p, a1s, cum_perm):
    """Returns (exec_price_per_share, cum_perm_after)"""
    ti = min(t, T_STEPS-1)
    ep = b1p[ti] - cum_perm
    b2 = b2p[ti] - cum_perm
    if q == 0:
        mid = (b1p[ti] + a1p[ti]) / 2.0 - cum_perm
        return mid, cum_perm
    cash = 0.0; rem = q
    q1 = min(rem, b1s[ti]); cash += q1 * ep; rem -= q1
    q2 = min(rem, b2s[ti]); cash += q2 * b2; rem -= q2
    if rem > 0:
        slip = 0.05 * (rem / INIT_INVENTORY)
        cash += rem * b2 * (1.0 - slip)
    return cash / q, cum_perm + GAMMA_PERM * q

def run_net_episode(net, b1p, b1s, b2p, b2s, a1p, a1s):
    inventory = INIT_INVENTORY; cum_perm = 0.0
    inv_path = [float(inventory)]
    price_path = []
    init_mid = (b1p[0] + a1p[0]) / 2.0
    for t in range(T_STEPS):
        if inventory <= 0:
            inv_path.append(0.0)
            price_path.append(np.nan)
            continue
        s = get_state(inventory, t, b1p, b1s, b2p, b2s, a1p, a1s, cum_perm)
        with torch.no_grad():
            a_idx = int(net(torch.from_numpy(s).unsqueeze(0).to(DEVICE)).argmax(1).item())
        q = int(round(inventory * a_idx / (N_ACTIONS - 1)))
        q = max(0, min(q, inventory))
        if q > 0:
            ep, cum_perm = execute_step(q, t, b1p, b1s, b2p, b2s, a1p, a1s, cum_perm)
        else:
            mid = (b1p[min(t, T_STEPS-1)] + a1p[min(t, T_STEPS-1)]) / 2.0 - cum_perm
            ep = mid
        price_path.append(ep / init_mid)
        inventory -= q
        inv_path.append(float(inventory))
    return np.array(inv_path), np.array(price_path)

def run_vwap_episode(b1p, b1s, b2p, b2s, a1p, a1s):
    inventory = INIT_INVENTORY; cum_perm = 0.0
    inv_path = [float(inventory)]
    price_path = []
    init_mid = (b1p[0] + a1p[0]) / 2.0
    t_pts = np.linspace(-1, 1, T_STEPS)
    weights = t_pts**2 + 0.5
    for t in range(T_STEPS):
        if inventory <= 0:
            inv_path.append(0.0); price_path.append(np.nan); continue
        w_rem = weights[t:].sum()
        frac = weights[t] / w_rem if w_rem > 0 else 1.0
        q = min(int(np.ceil(inventory * frac)), inventory)
        if q > 0:
            ep, cum_perm = execute_step(q, t, b1p, b1s, b2p, b2s, a1p, a1s, cum_perm)
        else:
            ep = (b1p[min(t, T_STEPS-1)] + a1p[min(t, T_STEPS-1)]) / 2.0 - cum_perm
        price_path.append(ep / init_mid)
        inventory -= q
        inv_path.append(float(inventory))
    return np.array(inv_path), np.array(price_path)

# Collect paths
ra_inv, ra_px = [], []
rn_inv, rn_px = [], []
vw_inv, vw_px = [], []

for tid in test_tids:
    lob = get_lob(tid)
    i, p = run_net_episode(ra_net, *lob); ra_inv.append(i); ra_px.append(p)
    i, p = run_net_episode(rn_net, *lob); rn_inv.append(i); rn_px.append(p)
    i, p = run_vwap_episode(*lob);        vw_inv.append(i); vw_px.append(p)

ra_inv = np.array(ra_inv); ra_px = np.array(ra_px)
rn_inv = np.array(rn_inv); rn_px = np.array(rn_px)
vw_inv = np.array(vw_inv); vw_px = np.array(vw_px)

t_norm_inv = np.linspace(0, 1, T_STEPS + 1)
t_norm_px  = np.linspace(0, 1, T_STEPS)

# ---------------------------------------------------------------------------
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fig.suptitle(
    f"RA-DQN vs RN-DQN vs VWAP — {len(test_tids)} OOS episodes (mean ± std)\n"
    f"Optiver LOB | T=10 | stock_id={sid}",
    fontweight='bold', fontsize=12)

# --- Left: inventory ---
ax = axes[0]
for arr, col, lbl in [
    (vw_inv, COL_VWAP, 'VWAP'),
    (ra_inv, COL_RA,   'RA-DQN'),
    (rn_inv, COL_RN,   'RN-DQN'),
]:
    mu = arr.mean(axis=0)
    sd = arr.std(axis=0)
    ax.plot(t_norm_inv, mu, color=col, lw=2.2, label=lbl)
    ax.fill_between(t_norm_inv, mu - sd, mu + sd, color=col, alpha=0.18)

ax.set_xlabel("Normalized time")
ax.set_ylabel("Shares")
ax.set_title("Inventory trajectory")
ax.legend(fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(-5, INIT_INVENTORY + 5)
ax.grid(True, alpha=0.25)

# --- Right: execution price ---
ax = axes[1]
for arr, col, lbl in [
    (vw_px, COL_VWAP, 'VWAP'),
    (ra_px, COL_RA,   'RA-DQN'),
    (rn_px, COL_RN,   'RN-DQN'),
]:
    # Nanmean across episodes at each step
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    ax.plot(t_norm_px, mu, color=col, lw=2.2, label=lbl)
    ax.fill_between(t_norm_px, mu - sd, mu + sd, color=col, alpha=0.18)

ax.set_xlabel("Normalized time")
ax.set_ylabel("Price (normalised by initial mid)")
ax.set_title("Execution price (per step)")
ax.legend(fontsize=11)
ax.set_xlim(0, 1)
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("trajectories_CI.png", dpi=300, bbox_inches='tight')
print("Saved trajectories_CI.png")
plt.close()
