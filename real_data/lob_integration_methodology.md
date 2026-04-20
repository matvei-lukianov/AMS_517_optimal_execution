# Methodology: The Hybrid Limit Order Book (LOB) Architecture

See also: [[optiver_results]] | [[../simulation/methodology]] | [[literature/reinforcement_learning_for_optimized_trade_execution_nevmyvaka|Nevmyvaka (2006)]] | [[literature/optimal_liquidation_robert_almgren_1998|Almgren & Chriss (1998)]] | [[literature/risk_averse_reinforcement_learning_for_algorithmic_trading_shen|Shen (2014)]]

To answer the core assignment objectives (comparing baseline RL against Risk-Averse RL over empirical datasets), we constructed an evaluation setup tracking the structural methods of *Nevmyvaka et al. (2006)* and *Shen et al. (2014)*, fused with the foundational mathematics of *Almgren & Chriss (1998)*.

## 1. Empirical Temporary Impact (The Optiver Dataset)

Instead of applying a synthetic proxy curve to compute the cost of crossing the spread, this simulation consumes raw high-frequency `parquet` snapshots from the top of the order book (Level 1 and Level 2) gathered from the Optiver dataset.

When the agent attempts to liquidate $q$ units at time $t$ in a sell-program:
1. **Level 1 Interaction**: Exhausts liquidity at the bid up to `bid_size1`. Execution price = `bid_price1`.
2. **Level 2 Execution (Walk the Book)**: If $q > \text{bid\_size1}$, remaining quota spills to `bid_price2`, consuming `bid_size2`.
3. **Deep-Book Penalty**: Any remainder exceeding Level 2 implies catastrophic liquidity exhaustion; linear degradation penalty representing deep-book walking.

This mirrors the empirical state representation used by Nevmyvaka (2006) exactly.

**Inventory calibration**: Q=200 units matches the empirical Optiver LOB depth
(median `bid_size1 + bid_size2 ≈ 200`), ensuring the agent operates near the
Level 1/2 boundary rather than always in the deep-book penalty zone.

## 2. Mathematical Permanent Impact (Almgren-Chriss, Eq 8)

Using purely historical datasets suffers from the "ghost trader problem" — the market does not dynamically shift fundamentally post-execution because the agent is replaying an isolated sequence of past events.

To solve this, we implement linear permanent impact from Almgren & Chriss (1998) Eq.8:

$$\text{Cumulative Permanent Shift} = \sum_{\tau \leq t} \gamma \cdot q_\tau$$

where γ = 2×10⁻⁵. After executing $q_t$ shares, this penalty is permanently
subtracted from all subsequent mid-price ticks in the current episode, blending
deterministic price decay with empirical LOB trajectories.

## 3. State Representation (Shen 2014)

Following Shen (2014) Sec.III-B, the state is:

| Component | Encoding | Dim |
|-----------|----------|-----|
| Inventory bucket | `inv_b = ⌊10·q/Q⌋`, clipped to [0,9] | 10 (one-hot) |
| Spread bucket | 3 buckets at 33/66 empirical quantiles per stock | 3 (one-hot) |
| Time bucket | 10 uniform buckets over T=10 steps | 10 (one-hot) |

Total state dim: 23.

## 4. T=10 Decision Points

Raw Optiver episodes span ~240 timesteps (10-min window at 250ms resolution).
Following Shen (2014) Sec.III-B, we subsample each episode to **T=10 evenly-spaced
decision points**, reducing training time while preserving the key inventory and
market dynamics.

## 5. Risk-Averse Update Rule

Standard DQN minimizes `(Q_target - Q)²`. The Shen (2014) risk-averse variant:

$$\mathcal{L} = \left[ u\!\left(\delta_t\right) \right]^2, \quad u(x) = \operatorname{sign}(x) |x|^{\lambda}$$

For λ=0.6 < 1, the concave utility amplifies large negative TD-errors and compresses
large positive ones. This shapes the value function to be pessimistic about downside
outcomes, producing risk-averse execution behavior.

See [[optiver_results]] for training convergence and evaluation metrics.
