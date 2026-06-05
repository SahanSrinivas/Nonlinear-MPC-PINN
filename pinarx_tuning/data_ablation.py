"""Limited-data ablation: train Res-Phys NARX on 200, 500, 1000, 2000 training
points and reproduce paper Fig 5 / Fig 6 / Table 4.

Usage:
  python data_ablation.py --device cuda --seed 0 --out-dir runs/data_ablation
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from data_gen import (gen_grid_train_val_split, gen_test1_set, gen_test2_set,
                        add_noise, SNR_T6_NOISY_LO, SNR_T6_NOISY_HI)
from resphys_narx import (ResPhysNARXHparams, ResPhysNARXModel)
from plots import (plot_fig5_style, plot_fig6_style, plot_test_case_4panel)
from train import PAPER_REF


# Paper Table 4 numbers (for the comparison table we print at the end)
PAPER_TABLE_4 = {
    2000: {"narx": {"t1": 0.001508, "t2": 0.01934},
            "pinarx": {"t1": 0.001242, "t2": 0.01556}},
    1000: {"narx": {"t1": 0.003266, "t2": 0.02677},
            "pinarx": {"t1": 0.001685, "t2": 0.01646}},
     500: {"narx": {"t1": 0.008336, "t2": 0.04664},
            "pinarx": {"t1": 0.004042, "t2": 0.02651}},
     200: {"narx": {"t1": 0.1368,   "t2": 0.2217},
            "pinarx": {"t1": 0.1433,   "t2": 0.2153}},
}


def train_one_size(n_points: int, args, t1, t2, noise: str | None
                     ) -> dict:
    """Train Res-Phys NARX with the given number of training points.
    Returns the model + its metrics."""
    print(f"\n[{n_points} points]  generating data...")
    tr, va = gen_grid_train_val_split(qf_levels=10, qc_levels=10,
                                          seed=args.seed,
                                          n_train_points=n_points)
    if noise is not None:
        snr_vec = SNR_T6_NOISY_LO if noise == "snr35" else SNR_T6_NOISY_HI
        tr = {**tr, "y": add_noise(tr["y"], snr_vec, seed=42)}
        va = {**va, "y": add_noise(va["y"], snr_vec, seed=43)}
    actual_size = tr["u"].shape[0]
    print(f"  train size = {actual_size}, val size = {va['u'].shape[0]}")

    hp = ResPhysNARXHparams(
        window=2, hidden=(200, 400, 200), activation="tanh",
        lr_adam=1e-3, lr_lbfgs=0.1,
        n_epochs_adam=args.epochs, lbfgs_iters=args.lbfgs,
        batch_size=64, weight_decay=0.0,
        early_stop_patience=120, seed=args.seed,
        include_u_lags=False, device=args.device,
        rk4_sub_steps=50, residual_l2=0.0,
    )
    t0 = time.time()
    model = ResPhysNARXModel(n_u=2, n_y=4, hp=hp)
    model.fit(tr, va, verbose=False)
    train_time_s = time.time() - t0

    t1_one = model.eval_mae_one_step(t1)
    t2_one = model.eval_mae_one_step(t2)
    t1_ar  = model.eval_mae(t1)
    t2_ar  = model.eval_mae(t2)
    print(f"  trained in {train_time_s:.0f}s  | T1 one-step = {t1_one:.4e}  "
            f"T2 one-step = {t2_one:.4e}")
    return {"model": model, "n_points": actual_size,
             "t1_one_step": float(t1_one), "t2_one_step": float(t2_one),
             "t1_autoreg": float(t1_ar),  "t2_autoreg": float(t2_ar),
             "train_time_s": train_time_s}


def render_table4(results: dict, noise: str | None) -> str:
    """Render the paper Table 4 style comparison."""
    lines = []
    if noise is None:
        lines.append("Paper Table 4 comparison (noiseless)")
    else:
        lines.append(f"Limited-data ablation ({noise})")
    lines.append("=" * 82)
    lines.append(f"  {'Data':<10}{'Test 1 (Within range)':>30}"
                   f"{'Test 2 (Extrapolation)':>30}")
    lines.append(f"  {'points':<10}"
                   f"{'NARX':>12}{'PI-NARX':>12}{'Res-Phys':>12}"
                   f"{'NARX':>12}{'PI-NARX':>12}{'Res-Phys':>12}")
    lines.append("  " + "-" * 80)
    for n in sorted(results.keys()):
        r = results[n]
        ref = PAPER_TABLE_4.get(n, {})
        narx_t1   = ref.get("narx",   {}).get("t1", float("nan"))
        narx_t2   = ref.get("narx",   {}).get("t2", float("nan"))
        pinarx_t1 = ref.get("pinarx", {}).get("t1", float("nan"))
        pinarx_t2 = ref.get("pinarx", {}).get("t2", float("nan"))
        lines.append(f"  {r['n_points']:<10}"
                       f"{narx_t1:>12.6f}{pinarx_t1:>12.6f}"
                       f"{r['t1_one_step']:>12.6f}"
                       f"{narx_t2:>12.6f}{pinarx_t2:>12.6f}"
                       f"{r['t2_one_step']:>12.6f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available()
                                                else "cpu")
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--sizes",      type=int, nargs="+",
                    default=[200, 500, 1000, 2000])
    ap.add_argument("--epochs",     type=int, default=1000)
    ap.add_argument("--lbfgs",      type=int, default=1000)
    ap.add_argument("--noise",      choices=[None, "snr35", "snr100"],
                    default=None)
    ap.add_argument("--out-dir",    default="runs/data_ablation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Limited-data ablation  noise={args.noise}  sizes={args.sizes}  "
            f"device={args.device} ===")
    t1, t2 = gen_test1_set(), gen_test2_set()
    if args.noise is not None:
        snr_vec = SNR_T6_NOISY_LO if args.noise == "snr35" else SNR_T6_NOISY_HI
        t1 = {**t1, "y": add_noise(t1["y"], snr_vec, seed=44)}
        t2 = {**t2, "y": add_noise(t2["y"], snr_vec, seed=45)}

    results = {}
    models_by_size = {}
    for n in args.sizes:
        r = train_one_size(n, args, t1, t2, noise=args.noise)
        results[r["n_points"]] = {k: v for k, v in r.items() if k != "model"}
        models_by_size[r["n_points"]] = r["model"]
        # Save partial JSON
        json.dump(results, open(out_dir / "results.json", "w"),
                  indent=2, default=str)

    # Fig 5 - C_A only, 2x2 grid per test case
    suffix = args.noise if args.noise else "noiseless"
    p5a, p5b = plot_fig5_style(models_by_size, t1, t2,
                                    str(out_dir / f"fig5a_test1_CA_{suffix}.png"),
                                    str(out_dir / f"fig5b_test2_CA_{suffix}.png"))
    # Fig 6 - all 4 outputs, one per test case
    p6a = plot_fig6_style(models_by_size, t1,
                              str(out_dir / f"fig6a_test1_all_{suffix}.png"),
                              test_label="(a) Test case 1 - interpolation")
    p6b = plot_fig6_style(models_by_size, t2,
                              str(out_dir / f"fig6b_test2_all_{suffix}.png"),
                              test_label="(b) Test case 2 - extrapolation")

    # Print the comparison table
    table_str = render_table4(results, args.noise)
    print()
    print(table_str)
    (out_dir / "table4.txt").write_text(table_str)

    print()
    print("Saved figures:")
    for name, path in [("fig5a", p5a), ("fig5b", p5b),
                          ("fig6a", p6a), ("fig6b", p6b)]:
        print(f"  {name:<8} -> {path}")


if __name__ == "__main__":
    main()
