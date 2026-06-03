"""Paper 3 Phase A bench: LLM-AutoOpt vs Optuna vs Random vs BO on PINN-MPC.

For each tuner, repeats:
  1. tuner.ask() -> proposed hparams
  2. train PINN with those hparams (reduced or full scale)
  3. evaluate trained model -> combined_score
  4. tuner.tell(cfg, score)
  5. log + checkpoint

Saves incremental results to disk so we can survive disconnects.

Usage (CPU smoke test):
  python -m agentic_pinn_mpc.bench --tuners random llm \
      --n-trials 3 --K1 200 --K2 200 --bs 50

Usage (RunPod full scale):
  python -m agentic_pinn_mpc.bench --tuners llm optuna random bo \
      --n-trials 25 --K1 10000 --K2 10000 --bs 100 \
      --output results/runpod_run1/
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .pinn_siso import (DEVICE, PINNHparams, train_pinn_siso)
from .evaluate import EvalScenario, evaluate_model
from .tuners import HSPACE, TUNERS, make_tuner


def load_training_data(data_dir: str | Path = None) -> tuple:
    """Load Kardamaki's pre-saved 1M-episode SISO training data."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "external"
    else:
        data_dir = Path(data_dir)
    p = data_dir / "siso_training_samples.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Clone https://github.com/ntua-unit-of-control"
            f"-and-informatics/pinn-mpc into {data_dir.parent}/external/.")
    data = torch.load(p, map_location="cpu", weights_only=False)
    return (data["x0_all"].to(DEVICE), data["u0_all"].to(DEVICE),
            data["ysp_all"].to(DEVICE), data["d0_all"].to(DEVICE))


def train_and_score(cfg: dict, x0_all, u0_all, ysp_all, d0_all,
                      K1: int, K2: int, bs: int,
                      n_eval_tracking: int = 200,
                      n_eval_disturbance: int = 200,
                      eval_seed: int = 42,
                      verbose: bool = False) -> dict:
    """One training+evaluation pass. Returns metrics dict."""
    hp = PINNHparams(
        w_ode=cfg["w_ode"], w_ic=cfg["w_ic"],
        w_ytrk=cfg["w_ytrk"], w_utrk=cfg["w_utrk"],
        w_du=cfg["w_du"], w_u=cfg["w_u"], w_x=cfg["w_x"],
        lr1=cfg["lr1"], lr2=cfg["lr2"],
        K1=K1, K2=K2, bs=bs,
    )
    t0 = time.time()
    try:
        model, hist = train_pinn_siso(
            hp, x0_all, u0_all, ysp_all, d0_all, verbose=verbose)
    except Exception as e:
        return {"score": 1e6, "error": str(e), "train_time": 0.0}
    train_time = time.time() - t0

    if hist.get("nan_at") is not None:
        return {"score": 1e6, "error": f"NaN at {hist['nan_at']}",
                "train_time": train_time,
                "final_loss_p1": float(hist["hist_p1"][-1]) if hist["hist_p1"] else float("nan"),
                "final_loss_p2": float("nan")}

    t0 = time.time()
    metrics = evaluate_model(model, n_tracking=n_eval_tracking,
                              n_disturbance=n_eval_disturbance,
                              seed=eval_seed,
                              use_kardamaki_samples=True)
    eval_time = time.time() - t0
    return {
        "score": metrics["combined_score"],
        "metrics": metrics,
        "train_time": train_time,
        "eval_time": eval_time,
        "final_loss_p1": float(hist["hist_p1"][-1]) if hist["hist_p1"] else float("nan"),
        "final_loss_p2": float(hist["hist_p2"][-1]) if hist["hist_p2"] else float("nan"),
        "n_epochs_p1": len(hist["hist_p1"]),
        "n_epochs_p2": len(hist["hist_p2"]),
    }


def run_tuner(tuner_name: str, n_trials: int,
                x0_all, u0_all, ysp_all, d0_all,
                K1: int, K2: int, bs: int,
                n_eval_tracking: int, n_eval_disturbance: int,
                seed: int, out_dir: Path) -> dict:
    """Run one tuner for n_trials. Saves checkpoint after each trial."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{tuner_name}_trials.json"

    print(f"\n=== Tuner: {tuner_name} (n_trials={n_trials}, seed={seed}) ===")
    print(f"  Training scale: K1={K1}, K2={K2}, bs={bs}")
    print(f"  Eval scale: {n_eval_tracking} tracking + "
          f"{n_eval_disturbance} disturbance episodes")

    tuner = make_tuner(tuner_name, seed=seed)
    trials = []
    best_score = float("inf")
    best_cfg = None
    t_run_start = time.time()
    for i in range(1, n_trials + 1):
        t0 = time.time()
        cfg = tuner.ask()
        result = train_and_score(
            cfg, x0_all, u0_all, ysp_all, d0_all,
            K1=K1, K2=K2, bs=bs,
            n_eval_tracking=n_eval_tracking,
            n_eval_disturbance=n_eval_disturbance,
        )
        score = result["score"]
        tuner.tell(cfg, score)
        if score < best_score:
            best_score, best_cfg = score, dict(cfg)
        elapsed = time.time() - t0
        trials.append({"iter": i, "cfg": cfg, "result": result,
                        "elapsed_s": elapsed})
        print(f"  iter {i:>2}/{n_trials}: score={score:.4f}  "
              f"best={best_score:.4f}  ({elapsed:.0f}s, "
              f"total {(time.time()-t_run_start)/60:.1f} min)")
        # Checkpoint after every trial
        with ckpt_path.open("w") as f:
            json.dump({"tuner": tuner_name, "n_trials": n_trials,
                        "best_score": best_score, "best_cfg": best_cfg,
                        "trials": trials,
                        "config": {"K1": K1, "K2": K2, "bs": bs,
                                    "n_eval_tracking": n_eval_tracking,
                                    "n_eval_disturbance": n_eval_disturbance,
                                    "seed": seed}},
                       f, indent=2, default=str)
    return {"name": tuner_name, "best_score": best_score,
            "best_cfg": best_cfg, "trials": trials}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuners", nargs="+",
                     default=["random", "bo", "optuna", "llm"],
                     choices=list(TUNERS.keys()))
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--K1", type=int, default=500,
                     help="Phase 1 epochs (Kardamaki uses 10000)")
    ap.add_argument("--K2", type=int, default=500,
                     help="Phase 2 epochs (Kardamaki uses 10000)")
    ap.add_argument("--bs", type=int, default=50,
                     help="Batch size (Kardamaki uses 100)")
    ap.add_argument("--n-eval-tracking", type=int, default=200,
                     help="Tracking episodes per eval (Kardamaki uses 3000)")
    ap.add_argument("--n-eval-disturbance", type=int, default=200,
                     help="Disturbance episodes per eval (Kardamaki uses 3000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default="results/phase_a_v1")
    ap.add_argument("--data-dir", type=str, default=None)
    a = ap.parse_args()

    print(f"=== Paper 3 Phase A bench ===")
    print(f"  Device: {DEVICE}")
    print(f"  Tuners: {a.tuners}")
    print(f"  N trials per tuner: {a.n_trials}")
    print(f"  Train scale: K1={a.K1}, K2={a.K2}, bs={a.bs}")
    print(f"  Eval scale: {a.n_eval_tracking}+{a.n_eval_disturbance} episodes")
    print(f"  Output dir: {a.output}")
    print()

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SISO training data...")
    x0_all, u0_all, ysp_all, d0_all = load_training_data(a.data_dir)
    print(f"  Loaded {x0_all.shape[0]} episodes")

    summary = {"config": vars(a), "device": str(DEVICE),
                "tuners": {}}
    for tuner_name in a.tuners:
        try:
            result = run_tuner(
                tuner_name, a.n_trials,
                x0_all, u0_all, ysp_all, d0_all,
                K1=a.K1, K2=a.K2, bs=a.bs,
                n_eval_tracking=a.n_eval_tracking,
                n_eval_disturbance=a.n_eval_disturbance,
                seed=a.seed, out_dir=out_dir)
            summary["tuners"][tuner_name] = {
                "best_score": result["best_score"],
                "best_cfg": result["best_cfg"],
                "n_trials": len(result["trials"]),
            }
        except Exception as e:
            print(f"  [{tuner_name}] FAILED: {e}")
            summary["tuners"][tuner_name] = {"error": str(e)}

    # Save final summary
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("=== FINAL SUMMARY ===")
    print(f"  {'tuner':<10}{'best score':>15}{'best cfg':>30}")
    for name, info in summary["tuners"].items():
        if "error" in info:
            print(f"  {name:<10}  ERROR: {info['error']}")
        else:
            cfg_str = f"w_ode={info['best_cfg']['w_ode']:.1f}, lr1={info['best_cfg']['lr1']:.2e}"
            print(f"  {name:<10}{info['best_score']:>15.4f}  ({cfg_str})")
    print(f"\nFull results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
