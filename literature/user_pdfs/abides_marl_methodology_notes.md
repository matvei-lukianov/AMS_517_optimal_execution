# ABIDES-MARL (arXiv:2511.02016) + slides — how they compare methods (for AMS517)

Sources: `2511.02016v1.pdf`, `ABIDES_MARL_Slides.pdf` in repo root; full-text extracts in this folder.

## What the stack is

- **ABIDES-MARL**: multi-agent RL on top of an event-driven LOB simulator (ABIDES), with a **StopSignalAgent** so several policies see a **synchronized** timestep (fixes ABIDES-Gym’s single-interrupt limitation).
- **Game**: Kyle-style multi-period market — informed trader, **liquidity trader** (execution), competing market makers, noise; equilibrium-like behavior approximated with **PPO** (independent policies per role).
- **Goal in the execution section**: study acquisition/liquidation when **liquidity and impact are endogenous** (MMs react), vs classical models with **fixed** impact.

## How they compare execution (takeaways for your write-up)

The paper explicitly contrasts **three** execution families for the liquidity trader (plus variants), then reports **implementation shortfall (IS)** in tables.

1. **PPO (multi-agent)** — liquidity trader trained **jointly** with other agents still learning. **Empirically weak**: opponents adapt and can exploit the liquidity trader’s pattern.
2. **Analytical schedule** — closed-form **linear trading intensity** \(\theta(n)\) from a **discrete-time, time-varying impact** extension of the single-agent problem (slides: backward recursion on \(\mu(n)\); “RL vs analytical” figure = **PPO vs this \(\theta(n)\) schedule** on inventory \(Q^{(n)}\) and price path \(p^{(n)}\)**).
3. **PPO-Single** — freeze informed + MM policies, **retrain only** the liquidity trader with PPO (~500 episodes in their tables). Often **best IS** under **nonlinear** MM quoting; interpretable as “post-optimization against fixed equilibrium-like opponents.”
4. **VWAP (trajectory-based)** — they build a schedule from **price–volume trajectories observed under the PPO evaluation** (not the same as your causal Optiver VWAP baseline).
5. **TWAP** — even inventory split; **most robust under linear** MM parameterization (they relate this to less information leakage when the environment is closer to “fixed dynamics”).

**Design lesson:** separate **(a)** classical / analytical **open-loop** schedules, **(b)** simple benchmarks (TWAP, VWAP-shaped), **(c)** RL trained in a **stationary** world (single-agent or fixed opponents), and **(d)** RL in a **non-stationary** multi-agent game — and report **all** with the **same** cost functional (they use **IS in dollars**; you use **bps / shortfall / CVaR** on real LOB).

## Slides vs your earlier figure

Slides p.7 label the twin panels as **remaining inventory \(Q^{(n)}\)** and **transaction price \(p^{(n)}\)** for **analytical vs PPO** under the **exogenous-impact** single-agent model — that is exactly the “publication-style trajectory” plot you wanted for intuition, before the MARL section.

## Mapping to AMS517 (Optiver, RA-DQN)

| Their ingredient | Your project analogue |
|------------------|----------------------|
| ABIDES-MARL simulator | **Historical Optiver LOB** episodes (already non-strategic “frozen” counterparties) |
| PPO liquidity trader | **RA-DQN / RN-DQN** |
| Analytical \(\theta(n)\) under known impact | **Not directly in real LOB** unless you fit a simple impact model on train data; **simulation** (`simulation/`) is the natural place for an AC-style analytical curve |
| TWAP / VWAP baselines | You already have **TWAP + causal VWAP** — closest to their Table 2–3 “TWAP / VWAP” rows |
| PPO-Single | **Fine-tune DQN** with frozen “world” (e.g. subset of stocks, or distilled LOB stats) — optional extension, not required for defense |
| IS in dollars | **Mean shortfall, bps, CVaR\(_{95}\)** on episode costs (Shen-style risk-averse utility is *your* extra dimension) |
| Trajectory figure | `optiver_agent_v3.py` now plots **mean ± std** paths vs **VWAP** on the **same** episodes — swap label to “analytical-style” only if you add a **fitted** schedule; otherwise keep “VWAP baseline” as honest |

## Citations (BibTeX-friendly)

```bibtex
@article{cheridito2025abidesmarl,
  title={ABIDES-MARL: A Multi-Agent Reinforcement Learning Environment for Endogenous Price Formation and Execution in a Limit Order Book},
  author={Cheridito, Patrick and Dupret, Jean-Loup and Wu, Zhexin},
  journal={arXiv preprint arXiv:2511.02016},
  year={2025}
}
```

## Optional next steps (if you want closer alignment)

- In **simulation**, overlay **Almgren–Chriss / closed-form** inventory path vs RL (they cite AC 2001 explicitly in the paper intro).
- In **real_data**, keep **TWAP / VWAP / RN** as baselines; add one sentence in the report that **multi-agent non-stationarity** is why joint MARL PPO can underperform — your setting is closer to **single-agent vs exogenous book** (like their Section 3 warm-up), so **RA-DQN vs VWAP** is a fair comparison story.
