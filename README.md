# Optimal Trade Execution via Risk-Averse Reinforcement Learning

This repository implements a Risk-Averse Deep RL framework for optimal liquidation, targeting **CVaR (Conditional Value at Risk)** reduction. The project bridges theoretical Gaussian simulations with empirical high-frequency Limit Order Book (LOB) data from Optiver.

See [[CLAUDE]] for session context and design decisions.

## Repository Structure

### `simulation/` (Phase 1–3)

Foundational testing of RL agents in synthetic Almgren-Chriss environments.

- `vwap_execution_simulation_revised.py` — Core simulation engine (Gaussian diffusion, square-root impact)
- [[simulation/methodology]] — MDP formulation, market impact model, policy descriptions
- [[simulation/results]] — Benchmarks: TWAP/VWAP/AC/RL CVaR (N=500 simulations)
- [[simulation/updated_paper]] — Theoretical paper draft (GCAPM framework)
- [[simulation/report]] — Summary report

### `real_data/` (Phase 5–7)

Empirical validation on the **Kaggle Optiver Realized Volatility** dataset.

- `optiver_agent.py` — GPU DQN implementation aligned with [[literature/risk_averse_reinforcement_learning_for_algorithmic_trading_shen|Shen (2014)]]
- `plot_execution.py` — Publication-quality plots (KDE, bar chart, per-episode scatter)
- [[real_data/optiver_results]] — Methodology table, training convergence, 4-policy comparison
- [[real_data/lob_integration_methodology]] — Hybrid LOB impact model (Nevmyvaka + Almgren-Chriss)

### `literature/`

Key references used to formulate the project:

- [[literature/optimal_liquidation_robert_almgren_1998|Almgren & Chriss (1998)]] — Mean-variance optimal liquidation, efficient frontier
- [[literature/reinforcement_learning_for_optimized_trade_execution_nevmyvaka|Nevmyvaka et al. (2006)]] — RL for trade execution, LOB state representation
- [[literature/risk_averse_reinforcement_learning_for_algorithmic_trading_shen|Shen (2014)]] — Risk-averse RL, utility transform `u(x) = sign(x)|x|^λ`
- [[literature/recent_advances_in_reinforcement_learning_in|Hambly et al.]] — RL in finance survey

### `optiver_data/`

Kaggle dataset parquet files. 20 stocks used for multi-stock training.

---

## Key Results

### Simulation (N=500)

| Policy | Mean Shortfall | CVaR₉₅ |
|--------|---------------:|--------:|
| TWAP | $6.49 | **$405.81** |
| RL CVaR (ours) | $61.37 | **$64.99** |

**84% CVaR reduction** vs TWAP. Full table: [[simulation/results]].

### Real Data (Optiver LOB, 20 stocks, 7,659 test episodes)

4-policy comparison (RA-DQN vs RN-DQN vs VWAP vs TWAP) per [[real_data/optiver_results]].
Aligned with Shen (2014) Table 1 structure.

---

## Design Principles

- **Risk-averse update** (Shen 2014): `loss = u(δ)²`, `u(x) = sign(x)|x|^λ`, λ=0.6
- **LOB-matched inventory**: Q=200 (median bid_size1+bid_size2 in Optiver data)
- **T=10 subsampling**: Each episode = 10 decision points (Shen Sec.III-B)
- **Multi-stock DQN**: GPU-accelerated, 20 stocks, 68,940 train episodes
- **Fair VWAP baseline**: Online causal U-curve (no lookahead)

---

*AMS517: Optimal Execution Research.*
