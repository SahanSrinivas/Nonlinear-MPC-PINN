"""Phase 2: Multi-objective Bayesian Optimization tuner.

Runs the MOBOTuner (NSGA-II via Optuna) over the same 9-D HSPACE as the
LLM tuner, but optimizing the 4 raw Kardamaki metrics directly instead of
a scalar composite. Produces a Pareto front of configs, then re-evaluates
each at full scale and reports the one with the best combined_score.

Usage:
  python phase2_mobo.py --n-trials 25 --K1 10000 --K2 10000 --bs 100

After completion, compare results/mobo_phaseA/ against
results/llm_phaseA/ for the paper's tuner ablation table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_pinn_mpc.bench import run_tuner, load_training_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--K1", type=int, default=10000)
    ap.add_argument("--K2", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--n-eval-tracking", type=int, default=500)
    ap.add_argument("--n-eval-disturbance", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="results/mobo_phaseA")
    a = ap.parse_args()

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2: Multi-objective BO bench ===")
    print(f"  Trials: {a.n_trials}, K1={a.K1}, K2={a.K2}, bs={a.bs}")
    print(f"  Eval: {a.n_eval_tracking}+{a.n_eval_disturbance} Kardamaki samples")
    print()

    print("Loading training data...")
    x0_all, u0_all, ysp_all, d0_all = load_training_data()
    print(f"  {x0_all.shape[0]} episodes")

    result = run_tuner(
        "mobo", a.n_trials,
        x0_all, u0_all, ysp_all, d0_all,
        K1=a.K1, K2=a.K2, bs=a.bs,
        n_eval_tracking=a.n_eval_tracking,
        n_eval_disturbance=a.n_eval_disturbance,
        seed=a.seed, out_dir=out_dir,
    )

    print()
    print("=== Phase 2 done ===")
    print(f"  Best combined_score: {result['best_score']:.4f}")
    print(f"  Best cfg: {result['best_cfg']}")
    print()
    print("Compare against results/llm_phaseA/ for tuner ablation.")


if __name__ == "__main__":
    main()
