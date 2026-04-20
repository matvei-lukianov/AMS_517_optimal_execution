# CLAUDE.md — Project Context for AMS517

> Auto-loaded on every session. Updated after major changes.

## Project

**AMS517: Optimal Trade Execution via Risk-Averse Reinforcement Learning**
Course project. Defense scheduled. See [[README]] for structure overview.

## Environment

- Python: `python3.9` (pyenv mts env — has torch, numpy, pandas)
- GPU: NVIDIA RTX 3080 Laptop, CUDA 11.7, torch 2.0.1+cu117
- Run scripts from their own directory: `cd real_data && python3.9 optiver_agent.py`
- Data: `optiver_data/book_train.parquet/stock_id=*/` — 112 stocks, parquet format

## Repository Map

```
AMS517/
├── simulation/          # Phase 1-3: Gaussian diffusion environment
│   ├── vwap_execution_simulation_revised.py   # main sim engine
│   ├── plot_simulation.py                      # publication plots
│   ├── results.md                              # simulation results + analysis
│   ├── methodology.md                          # MDP formulation
│   └── sim_outputs_revised/                    # PNG outputs
│
├── real_data/           # Phase 5-7: Optiver LOB empirical validation
│   ├── optiver_agent.py       # MAIN: GPU DQN, 4-policy eval
│   ├── plot_execution.py      # real-data plots (run after optiver_agent.py)
│   ├── optiver_results.md     # methodology + results writeup
│   └── optiver_results.csv    # output of optiver_agent.py
│
├── literature/          # Transcribed papers
│   ├── optimal_liquidation_robert_almgren_1998.md
│   ├── reinforcement_learning_for_optimized_trade_execution_nevmyvaka.md
│   └── risk_averse_reinforcement_learning_for_algorithmic_trading_shen.md
│
├── optiver_data/        # Kaggle Optiver Realized Volatility dataset
└── README.md
```

## Key Design Decisions

### Why DQN instead of tabular Q?
Tabular Q cannot generalize — unseen (inv, spread, time) states get zero Q-values.
DQN shares weights across states, trains on 20 stocks simultaneously.
This is what Nevmyvaka (2006) did (linear function approximator).

### Why inventory=200, not 2000?
Optiver LOB bid_size1 median≈100, L1+L2≈200. With inventory=2000,
every trade exceeds LOB depth → deep-book penalty dominates → no learning signal.

### Why T=10 steps per episode?
Shen (2014) Sec.III-B uses T=10. Also speeds up training 24x vs raw 240-step episodes.

### Why spread bucket instead of imbalance?
Shen (2014) explicitly tests spread vs no-spread and shows risk-averse RL
*responds to spread* (adjusts strategy) while risk-neutral RL does not.
This is the key mechanism of risk reduction.

### Why λ=0.6?
Shen (2014) Fig.2: switch point between risk-averse and risk-neutral behavior.
Below 0.6: too conservative (high mean cost). Above: converges to risk-neutral.

## Running the Code

```bash
# Simulation (Phase 1-3)
cd simulation
python3.9 vwap_execution_simulation_revised.py   # generates sim_outputs_revised/
python3.9 plot_simulation.py                     # 3 publication plots

# Real data (Phase 5-7)
cd real_data
python3.9 optiver_agent.py    # ~2h on RTX 3080, generates optiver_results.csv
python3.9 plot_execution.py   # 3 plots from optiver_results.csv
```

## Key Results

### Simulation (N=500, Gaussian diffusion)
| Policy | Mean Shortfall | CVaR₉₅ |
|--------|---------------:|--------:|
| TWAP | $6.49 | $405.81 |
| Static VWAP | $15.45 | $415.17 |
| AC Proxy | $20.63 | $211.77 |
| RL Risk-Neutral | $61.41 | $64.65 |
| **RL CVaR Dist.** | **$61.37** | **$64.99** |

→ **84% CVaR reduction** vs TWAP/VWAP at cost of higher mean shortfall.

### Real Data (20 stocks, 7,659 test episodes, Optiver LOB)
- RA-DQN trains to ~−10 bps avg reward (vs TWAP ~−30 bps)
- RA-DQN beats VWAP in **67% of episodes** on mean shortfall
- Full RA vs RN vs VWAP vs TWAP table: see [[real_data/optiver_results]]

## Literature Alignment

| Paper | What we use |
|-------|-------------|
| Almgren & Chriss (1998) | Impact model, efficient frontier concept |
| Nevmyvaka et al. (2006) | LOB state, Q-learning on real data, function approx |
| Shen et al. (2014) | Risk-averse utility u(δ)², λ=0.6, state design, T=10 |

**Dabney (QR-DQN 2017) is NOT in the assignment** — do not reference it.

## Common Mistakes Fixed

See [[real_data/optiver_results]] for full bug table. Key ones:
- `python3.9` not `python3` (torch lives in pyenv mts env)
- Always `cd real_data` before running (paths are relative)
- VWAP baseline must be causal (online U-curve, not episode-sum normalized)
