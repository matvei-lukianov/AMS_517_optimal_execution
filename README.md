# Optimal Trade Execution via Risk-Averse Reinforcement Learning

**AMS517 Course Project — Stony Brook University**
*Matvei Lukianov, John Walsh*

Implements a Risk-Averse Deep Q-Network (RA-DQN) for optimal liquidation on real Limit Order Book data, following [Shen et al. (2014)](literature/risk_averse_reinforcement_learning_for_algorithmic_trading_shen.md). Trained on 112 NASDAQ stocks from the [Optiver Realized Volatility dataset](https://kaggle.com/c/optiver-realized-volatility-prediction).

---

## Key Results (v3 — 112 stocks, 42,890 test episodes)

| Metric | RA-DQN | RN-DQN | VWAP | TWAP |
|--------|-------:|-------:|-----:|-----:|
| Mean cost (bps) | 23.52 | **20.20** | 23.44 | 23.34 |
| Std (bps) | 73.1 | 28.5 | 22.4 | 23.8 |
| **CVaR₉₅ (bps)** | 127.7 | **56.8** | 56.6 | 57.8 |
| Beats VWAP | **81%** | **81%** | — | — |

**RA vs RN CVaR gap: −70.9 bps** (Shen 2014 reports −10 to −15 bps — consistent direction, stronger effect).

---

## Method

The agent follows Shen (2014)'s **risk-sensitive TD update**:

```
Loss = E[u(δ)²],   u(x) = sign(x)|x|^λ,   λ = 0.6
```

For `λ < 1`, `u` is concave — large negative TD-errors are amplified, making the agent pessimistic about tail outcomes. CVaR₉₅ is used as an **evaluation metric only** (not the training objective).

### Extensions beyond Shen (2014)

| | Shen (2014) | This work |
|--|--|--|
| Agent | Tabular Q | GPU DQN |
| State | 23-d one-hot | **5 continuous features** |
| Stocks | 1 | **112** |
| Episodes | ~1k | **386k train** |

**State features:** inventory fraction, spread z-score, time fraction, LOB imbalance, book depth ratio.

**Market impact:** Nevmyvaka L1/L2 LOB sweep + Almgren-Chriss permanent impact (γ = 2×10⁻⁵).

---

## Repository Structure

```
real_data/
├── optiver_agent_v3.py      # Main: GPU DQN, RA + RN parallel training
├── optiver_agent_v2.py      # v2: with urgency penalty (less stable)
├── optiver_agent_v4.py      # v4: soft urgency (scale=0.5)
├── plot_trajectories.py     # Trajectory plots (inventory + price, mean±std)
├── plot_execution.py        # Summary plots from results CSV
├── lambda_sweep_v3.py       # λ sensitivity analysis
├── optiver_results_v3_latest.csv  # Full OOS results
├── optiver_results.md       # Methodology + results writeup
└── trajectories_CI.png      # Trajectory figure (used in presentation)

presentation/
├── main.tex                 # Beamer slides (~30 min)
└── figures/
    └── trajectories_CI.png

literature/
├── optimal_liquidation_robert_almgren_1998.md
├── reinforcement_learning_for_optimized_trade_execution_nevmyvaka.md
└── risk_averse_reinforcement_learning_for_algorithmic_trading_shen.md
```

---

## Reproducing Results

**Requirements:** Python 3.9, PyTorch 2.0 + CUDA 11.7, pandas, numpy.
**Data:** [Optiver Realized Volatility](https://kaggle.com/c/optiver-realized-volatility-prediction) → place at `optiver_data/book_train.parquet/`.

```bash
cd real_data
python3.9 optiver_agent_v3.py     # ~30 min on RTX 3080, saves dqn_ra/rn_v3_latest.pt
python3.9 plot_trajectories.py    # generates trajectories_CI.png
python3.9 plot_execution.py       # generates summary plots
```

**Key hyperparameters (v3):**

| Parameter | Value | Source |
|-----------|-------|--------|
| λ (risk aversion) | 0.6 | Shen Fig. 2 |
| T (steps/episode) | 10 | Shen Sec. III-B |
| Q (inventory) | 200 | Optiver LOB median L1+L2 |
| Episodes | 20,000 | — |
| Replay buffer | 200k | — |
| Batch size | 512 | — |
| Terminal penalty | 1.5% | — |

---

## Literature

- **Almgren & Chriss (1998)** — Mean-variance optimal liquidation, efficient frontier
- **Nevmyvaka, Feng & Kearns (2006)** — RL on real LOB data, beats VWAP on NASDAQ
- **Shen et al. (2014)** — Risk-averse RL via utility transform on TD-errors
