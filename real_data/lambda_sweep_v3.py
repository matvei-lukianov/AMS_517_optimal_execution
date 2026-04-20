#!/usr/bin/env python3.9
"""
Hyperparameter sweep for optiver_agent_v3.py (RA λ only; v1 untouched).

Each run: trains RA+RN on train-fit subset, evaluates on held-out val tail,
writes real_data/optiver_v3_last_metrics.json, exits early (no full OOS test).

Example (GPU, ~11 × shorter train — tune SWEEP_TRAIN_EPISODES):

  cd real_data
  SWEEP_TRAIN_EPISODES=5000 SWEEP_LAMBDA_STEP=0.05 python3.9 lambda_sweep_v3.py

  # Explicit lambdas (overrides MIN/MAX/STEP):
  SWEEP_LAMBDA_LIST=0.85,0.95,0.99 SWEEP_TRAIN_EPISODES=1500 python3.9 lambda_sweep_v3.py

Env passed through to optiver_agent_v3.py:
  SHEN_LAMBDA, OPTIVER_V3_TRAIN_EPISODES, OPTIVER_V3_VAL_FRAC,
  OPTIVER_V3_SWEEP_MODE=1
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(SCRIPT_DIR, "optiver_agent_v3.py")


def main():
    train_eps = os.environ.get("SWEEP_TRAIN_EPISODES", "6000")
    val_frac = os.environ.get("SWEEP_VAL_FRAC", "0.12")

    lam_list_raw = os.environ.get("SWEEP_LAMBDA_LIST", "").strip()
    if lam_list_raw:
        lambdas = [round(float(x.strip()), 4) for x in lam_list_raw.split(",") if x.strip()]
    else:
        lam_start = float(os.environ.get("SWEEP_LAMBDA_MIN", "0.4"))
        lam_end = float(os.environ.get("SWEEP_LAMBDA_MAX", "0.9"))
        lam_step = float(os.environ.get("SWEEP_LAMBDA_STEP", "0.05"))
        lambdas = []
        x = lam_start
        while x <= lam_end + 1e-9:
            lambdas.append(round(x, 4))
            x += lam_step

    rows = []
    for lam in lambdas:
        env = os.environ.copy()
        env["SHEN_LAMBDA"] = str(lam)
        env["OPTIVER_V3_TRAIN_EPISODES"] = train_eps
        env["OPTIVER_V3_VAL_FRAC"] = val_frac
        env["OPTIVER_V3_SWEEP_MODE"] = "1"
        env["OPTIVER_V3_RA_ONLY"] = os.environ.get("SWEEP_RA_ONLY", "1")
        print(f"\n=== λ={lam}  train_eps={train_eps}  val_frac={val_frac}  RA_ONLY={env['OPTIVER_V3_RA_ONLY']} ===\n", flush=True)
        r = subprocess.run(
            [sys.executable, V3],
            cwd=SCRIPT_DIR,
            env=env,
        )
        if r.returncode != 0:
            print(f"Run failed for λ={lam} (exit {r.returncode})", file=sys.stderr)
            sys.exit(r.returncode)
        metrics_path = os.path.join(SCRIPT_DIR, "optiver_v3_last_metrics.json")
        with open(metrics_path) as f:
            row = json.load(f)
        rows.append(row)

    out_name = os.environ.get("SWEEP_OUT_CSV", "lambda_sweep_v3_results.csv")
    out_csv = os.path.join(SCRIPT_DIR, os.path.basename(out_name))
    keys = [
        "ra_lambda",
        "train_episodes",
        "val_frac",
        "n_val_total",
        "n_val_eval",
        "ra_val_mean_bps",
        "ra_val_q95_bps",
        "rn_val_mean_bps",
        "rn_val_q95_bps",
        "run_id",
    ]  # RN columns optional when no RN weights
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})

    best = min(rows, key=lambda r: r["ra_val_q95_bps"])
    print("\n--- Sweep done ---")
    print(f"Wrote {out_csv}")
    print(
        f"Best RA by val q95 (bps): λ={best['ra_lambda']:.4f}  "
        f"q95={best['ra_val_q95_bps']:.2f}  mean={best['ra_val_mean_bps']:.2f}"
    )


if __name__ == "__main__":
    main()
