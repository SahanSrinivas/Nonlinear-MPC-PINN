"""Single entry point for training and evaluating Res-Phys NARX.

Used by:
  - CLI (this file's __main__) for one-off runs
  - The LEAN tuner / Optuna driver for HP search

The public function is `run_trial(hp_dict, ...)` which:
  1. Builds train/val/test data from the Thosar CSTR plant.
  2. Trains Res-Phys NARX with the given hyperparameters.
  3. Evaluates on Test Case 1 and Test Case 2 (paper-defined).
  4. Returns a dict with timing, val loss, and Test 1 / Test 2 MAE
     (both one-step and autoregressive), ready for the optimizer to read.

The default Test 1 / Test 2 schedules and the dense-grid training protocol
match the paper's plant exactly; we just train on more amplitudes than the
paper's APRBS sample drew.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_gen import (gen_grid_train_val_split, gen_test1_set, gen_test2_set,
                        add_noise, SNR_T6_NOISY_LO, SNR_T6_NOISY_HI)
from resphys_narx import (ResPhysNARXHparams, ResPhysNARXModel)


# Paper Table 2 reference numbers (we COMPARE against these; we don't try
# to reproduce them - see README)
PAPER_REF = {
    "narx_t1":    0.001508, "narx_t2":    0.01934,
    "pinarx_t1":  0.001242, "pinarx_t2":  0.01556,
}


def build_hparams(overrides: dict | None = None) -> ResPhysNARXHparams:
    """Build a ResPhysNARXHparams with optional overrides (the dict the
    LEAN tuner / Optuna will pass)."""
    if overrides:
        # `hidden` may come in as a list - convert to tuple
        if "hidden" in overrides and isinstance(overrides["hidden"], list):
            overrides["hidden"] = tuple(overrides["hidden"])
    return ResPhysNARXHparams(**(overrides or {}))


def run_trial(hp_overrides: dict | None = None,
                qf_levels: int = 10, qc_levels: int = 10,
                noise: str | None = None,    # None | "snr35" | "snr100"
                noise_seed: int = 42,
                verbose: bool = False,
                ) -> dict:
    """Train + evaluate one Res-Phys NARX configuration.

    Args:
      hp_overrides: dict of fields to override in ResPhysNARXHparams.
                    Any non-mentioned field keeps its default.
      qf_levels, qc_levels: dense-grid training resolution.
      noise: if 'snr35' or 'snr100', add Gaussian noise to TRAINING and
             TEST trajectories (paper Table 6 protocol). None = noiseless.
      noise_seed: RNG seed for the noise.

    Returns dict with metrics + provenance.
    """
    hp = build_hparams(hp_overrides)
    seed = hp.seed

    # --- Data ---
    tr, va = gen_grid_train_val_split(qf_levels=qf_levels, qc_levels=qc_levels,
                                          seed=seed)
    t1, t2 = gen_test1_set(), gen_test2_set()
    if noise is not None:
        snr_vec = SNR_T6_NOISY_LO if noise == "snr35" else SNR_T6_NOISY_HI
        # NOTE: we noise the OUTPUT trajectories used for windowing.
        # The plant is deterministic; "training/test" noise corresponds to
        # measurement noise.
        tr = {**tr, "y": add_noise(tr["y"], snr_vec, seed=noise_seed)}
        va = {**va, "y": add_noise(va["y"], snr_vec, seed=noise_seed + 1)}
        t1 = {**t1, "y": add_noise(t1["y"], snr_vec, seed=noise_seed + 2)}
        t2 = {**t2, "y": add_noise(t2["y"], snr_vec, seed=noise_seed + 3)}

    # --- Train ---
    t0 = time.time()
    model = ResPhysNARXModel(n_u=2, n_y=4, hp=hp)
    hist = model.fit(tr, va, verbose=verbose)
    train_time_s = time.time() - t0

    # --- Eval ---
    mae_t1_os = model.eval_mae_one_step(t1)
    mae_t2_os = model.eval_mae_one_step(t2)
    mae_t1_ar = model.eval_mae(t1)
    mae_t2_ar = model.eval_mae(t2)

    n_params = sum(p.numel() for p in model.net.parameters())

    return {
        "hp":           asdict(hp),
        "qf_levels":    qf_levels,
        "qc_levels":    qc_levels,
        "noise":        noise,
        "train_size":   int(tr["u"].shape[0]),
        "val_size":     int(va["u"].shape[0]),
        "n_params":     n_params,
        "train_time_s": train_time_s,
        "val_loss":     float(hist.get("val_loss_final",
                                            hist.get("best_val_adam",
                                                       float("nan")))),
        "mae_t1_one_step": float(mae_t1_os),
        "mae_t2_one_step": float(mae_t2_os),
        "mae_t1_autoreg":  float(mae_t1_ar),
        "mae_t2_autoreg":  float(mae_t2_ar),
        # primary OBJECTIVE for the optimizer: half the t1 MAE + half the t2
        # (the harder one) - rewards balanced interpolation+extrapolation.
        # Optimizer minimizes this.
        "objective":  float(0.5 * mae_t1_os + 0.5 * mae_t2_os),
        "paper_ref":  PAPER_REF,
    }


def print_summary(result: dict, save_to: str | None = None):
    print()
    print("=" * 72)
    print("Res-Phys NARX trial summary")
    print("=" * 72)
    print(f"  train size:    {result['train_size']}  "
            f"val size: {result['val_size']}  "
            f"params: {result['n_params']:,}")
    print(f"  noise:         {result['noise']}")
    print(f"  train time:    {result['train_time_s']:.1f}s")
    print(f"  val loss:      {result['val_loss']:.4e}")
    print()
    p = result["paper_ref"]
    print(f"  {'metric':<24}{'ours':>14}{'paper NARX':>14}{'paper PI-NARX':>16}")
    print(f"  {'-'*68}")
    print(f"  {'Test 1 one-step MAE':<24}"
            f"{result['mae_t1_one_step']:>14.6f}"
            f"{p['narx_t1']:>14.6f}{p['pinarx_t1']:>16.6f}")
    print(f"  {'Test 2 one-step MAE':<24}"
            f"{result['mae_t2_one_step']:>14.6f}"
            f"{p['narx_t2']:>14.6f}{p['pinarx_t2']:>16.6f}")
    print(f"  {'Test 1 autoreg  MAE':<24}"
            f"{result['mae_t1_autoreg']:>14.6f}{'-':>14}{'-':>16}")
    print(f"  {'Test 2 autoreg  MAE':<24}"
            f"{result['mae_t2_autoreg']:>14.6f}{'-':>14}{'-':>16}")
    print()
    print(f"  objective (optimizer minimizes): {result['objective']:.6e}")
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        with open(save_to, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  saved: {save_to}")


# ============================================================================
# CLI
# ============================================================================
def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                                            else "cpu")
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--qf-levels", type=int, default=10)
    ap.add_argument("--qc-levels", type=int, default=10)
    ap.add_argument("--noise",  choices=[None, "snr35", "snr100"], default=None)
    # Knobs the optimizer will sweep
    ap.add_argument("--window",     type=int,   default=2)
    ap.add_argument("--hidden",     nargs="+",  type=int,
                    default=[200, 400, 200])
    ap.add_argument("--activation", choices=["tanh", "relu", "gelu", "silu"],
                    default="tanh")
    ap.add_argument("--lr-adam",    type=float, default=1e-3)
    ap.add_argument("--lr-lbfgs",   type=float, default=0.1)
    ap.add_argument("--epochs",     type=int,   default=1000)
    ap.add_argument("--lbfgs",      type=int,   default=1000)
    ap.add_argument("--batch-size", type=int,   default=64)
    ap.add_argument("--residual-l2", type=float, default=0.0)
    ap.add_argument("--rk4-substeps", type=int, default=50)
    ap.add_argument("--out",        default=None, help="JSON path to save result")
    ap.add_argument("--verbose",    action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    overrides = dict(
        window=args.window,
        hidden=tuple(args.hidden),
        activation=args.activation,
        lr_adam=args.lr_adam,
        lr_lbfgs=args.lr_lbfgs,
        n_epochs_adam=args.epochs,
        batch_size=args.batch_size,
        lbfgs_iters=args.lbfgs,
        residual_l2=args.residual_l2,
        rk4_sub_steps=args.rk4_substeps,
        seed=args.seed,
        device=args.device,
    )
    print(f"Running Res-Phys NARX trial:  device={args.device}  seed={args.seed}  "
            f"noise={args.noise}")
    result = run_trial(hp_overrides=overrides,
                          qf_levels=args.qf_levels,
                          qc_levels=args.qc_levels,
                          noise=args.noise,
                          verbose=args.verbose)
    print_summary(result, save_to=args.out)
