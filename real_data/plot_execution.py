"""
Publication-quality plots for real-data (Optiver LOB) results.
Reads optiver_results.csv produced by optiver_agent.py.
Handles both old format (RL_CVaR_Shortfall/VWAP_Shortfall)
and new format (RA_DQN_bps, RN_DQN_bps, VWAP_bps, TWAP_bps).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
GRAY   = "#6B7280"


def load_results():
    df = pd.read_csv("optiver_results.csv")
    # Detect format
    if 'RA_DQN_bps' in df.columns:
        return df, 'new'
    else:
        return df, 'old'


def plot_distributions(df, fmt):
    fig, ax = plt.subplots(figsize=(10, 6))

    if fmt == 'new':
        policies = [
            ('RA_DQN_bps', 'Risk-Averse DQN (λ=0.6)', BLUE,   'solid'),
            ('RN_DQN_bps', 'Risk-Neutral DQN (λ=1.0)', RED,    'dashed'),
            ('VWAP_bps',   'VWAP Baseline',             ORANGE, 'dashed'),
            ('TWAP_bps',   'TWAP Baseline',             GRAY,   'dotted'),
        ]
    else:
        policies = [
            ('RL_bps' if 'RL_bps' in df.columns else 'RL_CVaR_Shortfall',
             'Risk-Averse DQN', BLUE, 'solid'),
            ('VWAP_bps' if 'VWAP_bps' in df.columns else 'VWAP_Shortfall',
             'VWAP Baseline', ORANGE, 'dashed'),
        ]

    for col, label, color, ls in policies:
        if col not in df.columns:
            continue
        data = df[col].values
        sns.kdeplot(data, ax=ax, color=color, linestyle=ls, linewidth=2,
                    label=f"{label}  μ={data.mean():.1f}  CVaR₉₅={np.quantile(data,0.95):.1f} bps",
                    fill=(ls == 'solid'), alpha=0.15)
        ax.axvline(np.quantile(data, 0.95), color=color, linestyle=':', linewidth=1.2, alpha=0.7)

    ax.set_title("Execution Cost Distribution: Optiver LOB\n"
                 "Risk-Averse vs Risk-Neutral DQN vs VWAP vs TWAP (Out-of-Sample)",
                 pad=14, fontweight='bold', fontsize=13)
    ax.set_xlabel("Execution Cost (bps, lower is better)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.savefig("Shortfall_Distribution.png", dpi=300)
    print("Saved Shortfall_Distribution.png")
    plt.close()


def plot_summary_bar(df, fmt):
    if fmt == 'new':
        policies = ['RA_DQN_bps', 'RN_DQN_bps', 'VWAP_bps', 'TWAP_bps']
        labels   = ['RA-DQN\n(λ=0.6)', 'RN-DQN\n(λ=1.0)', 'VWAP', 'TWAP']
        colors   = [BLUE, RED, ORANGE, GRAY]
    else:
        col_rl   = 'RL_bps' if 'RL_bps' in df.columns else 'RL_CVaR_Shortfall'
        col_vwap = 'VWAP_bps' if 'VWAP_bps' in df.columns else 'VWAP_Shortfall'
        policies = [col_rl, col_vwap]
        labels   = ['RA-DQN', 'VWAP']
        colors   = [BLUE, ORANGE]

    metrics = ['Mean\nCost (bps)', 'Std Dev\n(bps)', '95% CVaR\n(bps)']
    fns     = [lambda c: df[c].mean(), lambda c: df[c].std(),
               lambda c: np.quantile(df[c], 0.95)]

    x  = np.arange(len(metrics))
    w  = 0.8 / len(policies)
    offsets = np.linspace(-(len(policies)-1)/2, (len(policies)-1)/2, len(policies)) * w

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (col, lbl, clr) in enumerate(zip(policies, labels, colors)):
        if col not in df.columns:
            continue
        vals = [fn(col) for fn in fns]
        bars = ax.bar(x + offsets[i], vals, w*0.9, label=lbl, color=clr, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8,
                    color=clr, fontweight='bold')

    ax.set_title("Performance Summary: 4 Policies on Optiver LOB\n"
                 f"(20 stocks, out-of-sample, aligned with Shen 2014)",
                 pad=14, fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("Cost (bps) — Lower is Better", fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig("Summary_Bar_Chart.png", dpi=300)
    print("Saved Summary_Bar_Chart.png")
    plt.close()


def plot_scatter(df, fmt):
    if fmt == 'new':
        col_ra   = 'RA_DQN_bps'
        col_rn   = 'RN_DQN_bps'
        col_vwap = 'VWAP_bps'
        has_rn   = col_rn in df.columns
    else:
        col_ra   = 'RL_bps' if 'RL_bps' in df.columns else 'RL_CVaR_Shortfall'
        col_vwap = 'VWAP_bps' if 'VWAP_bps' in df.columns else 'VWAP_Shortfall'
        has_rn   = False

    fig, axes = plt.subplots(1, 2 if has_rn else 1, figsize=(14 if has_rn else 7, 6))
    if not has_rn:
        axes = [axes]

    for ax, (col_y, title_y) in zip(axes, [
        (col_ra,  'Risk-Averse DQN (λ=0.6)'),
        (col_rn,  'Risk-Neutral DQN (λ=1.0)') if has_rn else (col_ra, ''),
    ]):
        if col_y not in df.columns:
            continue
        ra   = df[col_y].values
        vwap = df[col_vwap].values
        lo   = np.percentile(np.concatenate([ra, vwap]), 2)
        hi   = np.percentile(np.concatenate([ra, vwap]), 98)
        c    = np.where(ra < vwap, BLUE, RED)
        ax.scatter(vwap, ra, c=c, alpha=0.25, s=6, edgecolors='none')
        ax.plot([lo, hi], [lo, hi], color=GRAY, linewidth=1.5, linestyle='--')
        pct = (ra < vwap).mean() * 100
        ax.text(0.05, 0.93, f"Beats VWAP: {pct:.0f}%",
                transform=ax.transAxes, fontsize=11, color=BLUE, fontweight='bold')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{title_y}\nvs VWAP (below diagonal = RL wins)",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("VWAP Cost (bps)", fontsize=11)
        ax.set_ylabel(f"{title_y.split('(')[0].strip()} Cost (bps)", fontsize=11)

    plt.suptitle("Per-Episode Scatter: RL Agents vs VWAP Baseline", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Per_Episode_Scatter.png", dpi=300)
    print("Saved Per_Episode_Scatter.png")
    plt.close()


if __name__ == "__main__":
    if not os.path.exists("optiver_results.csv"):
        print("optiver_results.csv not found — run optiver_agent.py first.")
        exit(1)

    df, fmt = load_results()
    print(f"Loaded {len(df)} episodes, format='{fmt}'")

    if fmt == 'new':
        for col, name in [('RA_DQN_bps','RA-DQN'), ('RN_DQN_bps','RN-DQN'),
                          ('VWAP_bps','VWAP'), ('TWAP_bps','TWAP')]:
            if col in df.columns:
                print(f"  {name}: mean={df[col].mean():.2f} bps, "
                      f"std={df[col].std():.2f}, CVaR95={np.quantile(df[col],0.95):.2f}")
    else:
        col = 'RL_bps' if 'RL_bps' in df.columns else 'RL_CVaR_Shortfall'
        print(f"  RL: mean={df[col].mean():.2f}, CVaR95={np.quantile(df[col],0.95):.2f}")

    plot_distributions(df, fmt)
    plot_summary_bar(df, fmt)
    plot_scatter(df, fmt)
    print("\nAll real-data plots saved.")
