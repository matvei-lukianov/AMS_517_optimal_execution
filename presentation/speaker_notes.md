# Speaker Notes — Risk-Averse DQN onwards (~15 min)

---

## Slide: Risk-Averse DQN — Shen (2014) Update Rule (~2.5 min)

Now we get to the core technical contribution — how Shen (2014) actually introduces risk aversion into Q-learning.

In standard DQN we minimize the squared TD-error: L = E[δ²]. This is a risk-neutral agent — it treats upside and downside surprises symmetrically.

Shen's idea is to replace δ with u(δ), where u(x) = sign(x)·|x|^λ. So instead of minimizing δ², we minimize u(δ)².

Key observation: u(δ)² = |δ|^(2λ) — the sign cancels out. The loss is **symmetric** in δ. There is no downside asymmetry in the strict sense.

What λ=0.6 actually does: it makes the loss sub-quadratic. For large |δ| — whether positive or negative — the penalty grows slower than quadratic. For small |δ|, near zero, the gradient is actually larger than in the standard case. Look at the plot: the blue curve is steeper near zero and flatter at the extremes compared to the red line.

The consequence for Q-values: extreme TD-errors in either direction get compressed. Large positive experiences don't inflate Q-values as aggressively. The agent ends up with more **conservative, pessimistic estimates** — it doesn't get over-excited by lucky episodes. That conservatism is the mechanism behind the risk-averse behavior.

One important clarification, highlighted in the red box: CVaR₉₅ in our work is an **evaluation metric, not the training objective**. The utility transform acts on TD-errors inside Q-learning, not on the return distribution directly. Direct tail optimization would require distributional RL — that's future work.

---

## Slide: Extensions Beyond Shen (2014) (~2 min)

Now let's talk about what we changed relative to the original paper.

First — we use a Deep Q-Network instead of tabular Q-learning. Shen works with one stock and roughly a thousand episodes. A tabular agent cannot generalize: if a state (inventory, spread, time) wasn't seen during training, its Q-value is simply zero — which means a random policy. DQN solves this by sharing weights across similar states. We train on 112 stocks and 386,000 episodes.

Second — **continuous state** instead of a 23-dimensional one-hot vector. Shen discretizes inventory into 10 buckets, spread into 3, time into 10. We feed raw continuous values: inventory fraction, spread z-score, time fraction, LOB imbalance, and book depth. No bucket boundary artifacts, and two additional features that Shen doesn't use at all.

The blue block at the bottom shows what we kept unchanged: the utility transform with λ=0.6, T=10 steps per episode, reward in basis points, and the core state variables.

---

## Slide: Dataset — Optiver Realized Volatility Prediction (~2 min)

The data is from a 2021 Kaggle competition by Optiver — QR code on the left. The original goal of the competition was to predict realized volatility. We don't care about that — we just use the order book snapshots as a realistic market environment to trade in.

Concretely: the dataset gives us real LOB data for 112 NASDAQ stocks. Each "episode" in our framework is one 10-minute trading window. Inside that window we have per-second snapshots of the order book — best bid and ask prices, and the available quantity at the first two price levels.

We subsample each window down to 10 decision points, as Shen does, so the agent makes one sell decision every minute on average.

Now, why Q=200? Look at the table on the right — bid_size1 is the number of shares available at the best bid. In this dataset the median is around 100 shares, so levels 1 and 2 together give roughly 200 shares of depth. We set our starting inventory to match that. If we set Q=2000 instead, every single trade would blow through both levels and hit the deep book, where we apply a big penalty. The agent can't learn anything meaningful from a signal that's always maxed out. Q=200 keeps trades in the realistic range where the LOB structure actually matters.

We use 90% of episodes for training and 10% for testing — that's 386k and 43k episodes respectively.

---

## Slide: Implementation (~1.5 min)

The network architecture is straightforward: 5 input features, two hidden layers of 128 neurons with ReLU, and 11 outputs — one Q-value per discrete action from 0% to 100% of remaining inventory.

We trained on an RTX 3080. One engineering detail worth mentioning: RA-DQN and RN-DQN were trained **in parallel on two threads** sharing the same GPU — this gave roughly 50% speedup versus sequential training.

The terminal penalty of 1.5% on unexecuted inventory is a critical design parameter. Without it the agent learns to do nothing, because every trade has negative reward. The penalty forces it to sell.

The table on the right traces every design choice back to a specific paper and section.

---

## Slide: Execution Trajectories (~2 min)

Now results. This plot shows 383 out-of-sample episodes for stock id=0. Two subplots: left is inventory over time, right is execution price at each step normalized by the initial mid-price.

Look at the left panel. Blue — RA-DQN — sells aggressively early: by the midpoint inventory is nearly zero. Red — RN-DQN — sells more uniformly. Green — VWAP — follows a fixed U-shaped curve, heavier at the start and end.

On the right panel, RA-DQN gets slightly better prices early in the episode — it sells before its own permanent impact accumulates. The shaded bands are one standard deviation across episodes.

---

## Slide: Out-of-Sample Results (~2.5 min)

Now the numbers. 112 stocks, 42,890 test episodes.

On mean cost, RN-DQN is the best: 20.2 basis points versus 23.4 for VWAP — a 14% reduction. The standard error here is 0.14 versus 0.11, so the gap is roughly 13 standard errors — statistically unambiguous. Both RL agents beat VWAP in 81% of episodes.

On CVaR₉₅ — the average cost in the worst 5% of outcomes — the picture is more troubling. RN-DQN and VWAP are almost identical: 56.8 versus 56.6. RA-DQN shows 127.7. So RA is actually **worse** than RN by +70.9 bps.

This is the opposite of what Shen finds. In the paper, the risk-averse agent reduces CVaR by 10 to 15 bps relative to the risk-neutral one. We get the reverse. The direction is wrong, not just the magnitude.

This is the key honest result of the paper: the TD-error utility trick — replacing δ with u(δ) — does not translate into actual tail improvement on real LOB data. It's a surrogate, and on this dataset it misfires. That's precisely what motivates distributional RL as the next step: QR-DQN or IQN would optimize the return distribution directly, not through a proxy.

---

## Slide: Summary & Future Work (~1.5 min)

To summarize. We extended Shen (2014) to real LOB data: GPU DQN instead of tabular Q, 112 stocks, continuous state, parallel training.

On mean cost, the results are positive: both RL agents beat VWAP in 81% of episodes, and RN-DQN's 14% improvement is statistically significant at 13 standard errors.

On CVaR, the results are honest and negative: RA-DQN is +70.9 bps worse than RN-DQN, opposite to what Shen reports. The TD-error utility proxy does not translate to tail improvement on real data. That's not a failure of the project — it's the main finding: the surrogate is insufficient, and direct tail optimization via distributional RL is the necessary next step.

Four directions forward: QR-DQN/IQN for direct CVaR optimization, multiple seeds for confidence intervals, actor-critic with continuous actions, and a richer state including realized volatility and multi-level imbalance.
