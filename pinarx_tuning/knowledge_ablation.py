"""Limited-knowledge ablation: train Res-Phys NARX with four levels of
physics knowledge and reproduce paper Table 5 (equivalent).

  physics_mode = full   -> mass + energy + level (all 4 channels)
  physics_mode = mass   -> mass + level (C_A, h);  NN learns T, T_c
  physics_mode = energy -> energy (T, T_c);        NN learns C_A, h
  physics_mode = none   -> pure NARX (no physics)

This is the apples-to-apples test for "how much does the physics inductive
bias buy you?" — identical architecture / data / optimizer, only the
physics step changes.

Usage:
  python knowledge_ablation.py --device cuda --seed 0 \
      --out-dir runs/knowledge_ablation
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from data_gen import (gen_grid_train_val_split, gen_test1_set, gen_test2_set,
                        add_noise, NOISE_PROFILES)
from resphys_narx import (ResPhysNARXHparams, ResPhysNARXModel)
from train import PAPER_REF


PHYSICS_MODES = ("full", "mass", "energy", "none")

MODE_LABEL = {
    "full":   "Full physics (mass+energy+level)",
    "mass":   "Mass + level only  (C_A, h)",
    "energy": "Energy only        (T, T_c)",
    "none":   "No physics  (pure NARX)",
}


def train_one_mode(physics_mode: str, args, tr, va, t1, t2) -> dict:
    print(f"\n[physics_mode={physics_mode}]")
    hp = ResPhysNARXHparams(
        window=2, hidden=(200, 400, 200), activation="tanh",
        lr_adam=1e-3, lr_lbfgs=0.1,
        n_epochs_adam=args.epochs, lbfgs_iters=args.lbfgs,
        batch_size=64, weight_decay=0.0,
        early_stop_patience=120, seed=args.seed,
        include_u_lags=False, device=args.device,
        rk4_sub_steps=50, residual_l2=0.0,
        physics_mode=physics_mode,
    )
    t0 = time.time()
    model = ResPhysNARXModel(n_u=2, n_y=4, hp=hp)
    model.fit(tr, va, verbose=False)
    train_time_s = time.time() - t0

    t1_one = model.eval_mae_one_step(t1)
    t2_one = model.eval_mae_one_step(t2)
    t1_ar  = model.eval_mae(t1)
    t2_ar  = model.eval_mae(t2)
    print(f"  trained in {train_time_s:.0f}s  | "
            f"T1 one-step = {t1_one:.4e}  T2 one-step = {t2_one:.4e}")
    return {
        "physics_mode":  physics_mode,
        "label":         MODE_LABEL[physics_mode],
        "t1_one_step":   float(t1_one),
        "t2_one_step":   float(t2_one),
        "t1_autoreg":    float(t1_ar),
        "t2_autoreg":    float(t2_ar),
        "train_time_s":  train_time_s,
    }


def render_table5(results: dict) -> str:
    lines = []
    lines.append("Table 5 (equivalent) - Limited physics knowledge "
                  "ablation (noiseless, 2000 train pts)")
    lines.append("=" * 92)
    lines.append(f"  {'Physics knowledge':<38}"
                   f"{'T1 one-step':>14}{'T2 one-step':>14}"
                   f"{'T1 autoreg':>14}{'T2 autoreg':>14}")
    lines.append("  " + "-" * 92)
    for mode in PHYSICS_MODES:
        if mode not in results:
            continue
        r = results[mode]
        lines.append(f"  {r['label']:<38}"
                       f"{r['t1_one_step']:>14.6f}"
                       f"{r['t2_one_step']:>14.6f}"
                       f"{r['t1_autoreg']:>14.6f}"
                       f"{r['t2_autoreg']:>14.6f}")
    lines.append("")
    lines.append(f"  Paper reference (full physics): "
                   f"NARX T1={PAPER_REF['narx_t1']:.6f} T2={PAPER_REF['narx_t2']:.6f}  |  "
                   f"PI-NARX T1={PAPER_REF['pinarx_t1']:.6f} T2={PAPER_REF['pinarx_t2']:.6f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available()
                                            else "cpu")
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--modes",   nargs="+",
                    default=list(PHYSICS_MODES),
                    choices=list(PHYSICS_MODES))
    ap.add_argument("--epochs",  type=int, default=1000)
    ap.add_argument("--lbfgs",   type=int, default=1000)
    ap.add_argument("--noise",   choices=[None, "snr35", "snr100", "snr250"],
                    default=None)
    ap.add_argument("--out-dir", default="runs/knowledge_ablation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Limited-knowledge ablation  modes={args.modes}  "
            f"noise={args.noise}  device={args.device} ===")

    tr, va = gen_grid_train_val_split(qf_levels=10, qc_levels=10,
                                          seed=args.seed)
    t1, t2 = gen_test1_set(), gen_test2_set()
    if args.noise is not None:
        snr_vec = NOISE_PROFILES[args.noise]
        tr = {**tr, "y": add_noise(tr["y"], snr_vec, seed=42)}
        va = {**va, "y": add_noise(va["y"], snr_vec, seed=43)}
        t1 = {**t1, "y": add_noise(t1["y"], snr_vec, seed=44)}
        t2 = {**t2, "y": add_noise(t2["y"], snr_vec, seed=45)}
    print(f"  train={tr['u'].shape[0]}  val={va['u'].shape[0]}")

    results = {}
    for mode in args.modes:
        r = train_one_mode(mode, args, tr, va, t1, t2)
        results[mode] = r
        json.dump(results, open(out_dir / "results.json", "w"),
                  indent=2, default=str)

    table_str = render_table5(results)
    print()
    print(table_str)
    (out_dir / "table5.txt").write_text(table_str)
    print()
    print(f"  Saved -> {out_dir / 'results.json'}")
    print(f"  Saved -> {out_dir / 'table5.txt'}")


if __name__ == "__main__":
    main()
