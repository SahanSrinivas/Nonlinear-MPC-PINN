"""Phase 3: LEAN-LLM-OPT 3-agent tuner.

Same 9-D HSPACE as LLM/MOBO/BO, but the search strategy is:
  - Diagnostic agent: classifies the worst metric failure mode
  - Strategy agent:   retrieves playbook fix from ref_pinn_fixes.yaml
  - Tuning agent:     applies multipliers to the current best config

Unlike the LLM tuner, this needs PER-METRIC feedback (not just a scalar
score) - bench.py passes metrics through tell().

Unlike MOBO, it uses ENGINEERING DOMAIN KNOWLEDGE (the playbook) rather than
black-box search. The expected payoff: faster convergence (5-8 trials) on the
SAME hardware budget vs LLM (20-25 trials).

Usage:
  python phase3_lean.py --n-trials 15 --K1 10000 --K2 10000 --bs 100
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from agentic_pinn_mpc.bench import load_training_data
from agentic_pinn_mpc.pinn_siso import PINNHparams, train_pinn_siso
from agentic_pinn_mpc.evaluate import evaluate_model
from agentic_pinn_mpc.lean_tuner import LeanTuner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--K1", type=int, default=10000)
    ap.add_argument("--K2", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--n-eval-tracking", type=int, default=500)
    ap.add_argument("--n-eval-disturbance", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="results/lean_phaseA")
    a = ap.parse_args()

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "lean_trials.json"

    print("=== Phase 3: LEAN 3-agent bench ===")
    print(f"  Trials: {a.n_trials}, K1={a.K1}, K2={a.K2}, bs={a.bs}")
    print(f"  Eval: {a.n_eval_tracking}+{a.n_eval_disturbance} Kardamaki samples")
    print()

    print("Loading training data...")
    x0_all, u0_all, ysp_all, d0_all = load_training_data()
    print(f"  {x0_all.shape[0]} episodes\n")

    tuner = LeanTuner(seed=a.seed)
    trials = []
    best_score = float("inf")
    best_cfg = None
    best_metrics = None
    t_start = time.time()

    for i in range(1, a.n_trials + 1):
        t0 = time.time()
        cfg_with_meta = tuner.ask()
        diagnosis_meta = cfg_with_meta.pop("__lean_diagnosis__", None)
        cfg = cfg_with_meta
        train_changes = tuner.pop_train_changes()

        # Build PINNHparams with optional train_changes (e.g. importance alpha)
        hp_kwargs = {k: v for k, v in cfg.items()
                     if k in PINNHparams.__dataclass_fields__}
        for k, v in train_changes.items():
            if k in PINNHparams.__dataclass_fields__:
                hp_kwargs[k] = v
        hp = PINNHparams(K1=a.K1, K2=a.K2, bs=a.bs, **hp_kwargs)

        try:
            model, hist = train_pinn_siso(hp, x0_all, u0_all, ysp_all, d0_all,
                                            verbose=False)
            if hist.get("nan_at") is not None:
                raise RuntimeError(f"NaN at {hist['nan_at']}")
            metrics = evaluate_model(model, n_tracking=a.n_eval_tracking,
                                       n_disturbance=a.n_eval_disturbance,
                                       use_kardamaki_samples=True)
            score = metrics["combined_score"]
        except Exception as e:
            metrics = {}
            score = 1e6
            print(f"  trial {i} failed: {e}")

        tuner.tell(cfg, score, metrics=metrics)
        if score < best_score:
            best_score, best_cfg, best_metrics = score, dict(cfg), dict(metrics)
        elapsed = time.time() - t0

        diag_str = ""
        if diagnosis_meta:
            diag_str = (f"  diag={diagnosis_meta['failure_mode']}/"
                        f"{diagnosis_meta['severity']} "
                        f"(ratio={diagnosis_meta['worst_ratio']:.2f})")
        print(f"  iter {i:>2}/{a.n_trials}: score={score:.4f}  "
              f"best={best_score:.4f}{diag_str}  "
              f"({elapsed:.0f}s, total {(time.time()-t_start)/60:.1f} min)")

        trials.append({
            "iter": i,
            "cfg": cfg,
            "diagnosis": diagnosis_meta,
            "train_changes": train_changes,
            "metrics": metrics,
            "score": score,
            "elapsed_s": elapsed,
        })
        with open(ckpt_path, "w") as f:
            json.dump({
                "tuner": "lean",
                "n_trials": a.n_trials,
                "best_score": best_score,
                "best_cfg": best_cfg,
                "best_metrics": best_metrics,
                "trials": trials,
            }, f, indent=2, default=str)

    print()
    print("=== Phase 3 done ===")
    print(f"  Best combined_score: {best_score:.4f}")
    print(f"  Best cfg: {best_cfg}")
    print(f"  Best metrics: {best_metrics}")
    print()
    print(f"Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
