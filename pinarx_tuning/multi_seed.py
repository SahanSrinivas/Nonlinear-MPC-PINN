"""multi_seed.py - run NARX and PI-NARX on N seeds, paper-exact protocol.

For each seed:
  - Generate the 5000-min trajectory (paper recipe), split 2000/3000
  - Generate Test Case 1, Test Case 2 (paper exact)
  - Train NARX (Adam 1000 + L-BFGS 1000)
  - Train PI-NARX (Adam 1000 + L-BFGS, lambda_l=1e10, lambda_p=0.01)
  - Eval one-step + autoregressive MAE on both tests

Output:
  - runs/multi_seed_results.json  (every seed's numbers, paper hyperparams)
  - runs/multi_seed_summary.txt   (mean +/- std table, side-by-side w/ paper)

Usage:
  python multi_seed.py --device cuda --seeds 0 1 2 3 4 --pinarx-lbfgs 9000
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from data_gen import (gen_train_val_split, gen_grid_train_val_split,
                         gen_test1_set, gen_test2_set)
from narx import (NARXHparams, NARXModel)
from pi_narx import (PINARXHparams, PINARXModel)


PAPER = {"narx_t1":    0.001508, "narx_t2":    0.01934,
          "pinarx_t1":  0.001242, "pinarx_t2":  0.01556}


def run_one_seed(seed: int, device: str,
                  protocol: str = "grid",
                  pinarx_lbfgs: int = 9000,
                  narx_lbfgs:   int = 1000,
                  epochs: int = 1000,
                  verbose: bool = False) -> dict:
    print(f"\n{'='*70}\n  SEED {seed}  (device={device}, protocol={protocol})\n{'='*70}")

    if protocol == "grid":
        tr, va = gen_grid_train_val_split(qf_levels=10, qc_levels=10, seed=seed)
    else:
        tr, va = gen_train_val_split(N_total=5000, N_train=2000, seed=seed)
    t1, t2 = gen_test1_set(), gen_test2_set()

    out = {"seed": seed, "device": device,
            "train_amplitudes": sorted(set(map(tuple, tr["u"].tolist())))}

    # -------- NARX --------
    t0 = time.time()
    hp = NARXHparams(window=2, hidden=(200, 400, 200), activation="tanh",
                       lr_adam=1e-3, n_epochs_adam=epochs, batch_size=64,
                       lbfgs_iters=narx_lbfgs, weight_decay=0.0,
                       early_stop_patience=120, seed=seed,
                       include_u_lags=False, device=device)
    m_narx = NARXModel(n_u=2, n_y=4, hp=hp)
    m_narx.fit(tr, va, verbose=verbose)
    out["narx_t1_one_step"] = m_narx.eval_mae_one_step(t1)
    out["narx_t2_one_step"] = m_narx.eval_mae_one_step(t2)
    out["narx_t1_autoreg"]  = m_narx.eval_mae(t1)
    out["narx_t2_autoreg"]  = m_narx.eval_mae(t2)
    out["narx_time_s"]      = time.time() - t0
    print(f"  NARX     Test1 one-step = {out['narx_t1_one_step']:.6f}  "
            f"Test2 one-step = {out['narx_t2_one_step']:.6f}  "
            f"({out['narx_time_s']:.0f}s)")

    # -------- PI-NARX --------
    t0 = time.time()
    pp = PINARXHparams(window=2, hidden=(200, 400, 200), activation="tanh",
                          lr_adam=1e-3, n_epochs_adam=epochs, batch_size=64,
                          lbfgs_iters=pinarx_lbfgs, weight_decay=0.0,
                          early_stop_patience=120, seed=seed,
                          lambda_data=1e10, lambda_phys=1e-2,
                          n_collocation=10_000, dt_phys=1.0,
                          include_u_lags=False, device=device)
    m_pi = PINARXModel(n_u=2, n_y=4, hp=pp)
    m_pi.fit(tr, va, verbose=verbose)
    out["pinarx_t1_one_step"] = m_pi.eval_mae_one_step(t1)
    out["pinarx_t2_one_step"] = m_pi.eval_mae_one_step(t2)
    out["pinarx_t1_autoreg"]  = m_pi.eval_mae(t1)
    out["pinarx_t2_autoreg"]  = m_pi.eval_mae(t2)
    out["pinarx_time_s"]      = time.time() - t0
    print(f"  PI-NARX  Test1 one-step = {out['pinarx_t1_one_step']:.6f}  "
            f"Test2 one-step = {out['pinarx_t2_one_step']:.6f}  "
            f"({out['pinarx_time_s']:.0f}s)")
    return out


def mean_std(xs):
    if len(xs) <= 1:
        return float(xs[0]) if xs else float("nan"), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def summarize(results: list[dict]) -> str:
    keys = ["narx_t1_one_step", "narx_t2_one_step",
             "pinarx_t1_one_step", "pinarx_t2_one_step"]
    paper = {"narx_t1_one_step":   PAPER["narx_t1"],
              "narx_t2_one_step":   PAPER["narx_t2"],
              "pinarx_t1_one_step": PAPER["pinarx_t1"],
              "pinarx_t2_one_step": PAPER["pinarx_t2"]}
    lines = ["", "=" * 76, "Multi-seed summary  (paper-exact protocol)",
              "=" * 76,
              f"  {'metric':<26}{'mean':>11}{'std':>11}{'paper':>11}"
                f"{'vs paper':>15}",
              "  " + "-" * 74]
    for k in keys:
        vals = [float(r[k]) for r in results]
        m, s = mean_std(vals)
        p = paper[k]
        ratio = m / p
        lines.append(f"  {k:<26}{m:>11.6f}{s:>11.6f}{p:>11.6f}"
                       f"{ratio:>15.2f}x")
    lines.append("")
    lines.append(f"  N seeds: {len(results)}")
    lines.append(f"  Seeds: {[r['seed'] for r in results]}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds",  type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--protocol", choices=["grid", "paper-aprbs"], default="grid",
                    help="'grid' = 10x10 dense (Q_f, Q_c) grid (canonical, reproducible). "
                         "'paper-aprbs' = paper's exact 5000-min APRBS (seed-dependent).")
    ap.add_argument("--pinarx-lbfgs", type=int, default=9000)
    ap.add_argument("--narx-lbfgs",   type=int, default=1000)
    ap.add_argument("--epochs",       type=int, default=1000)
    ap.add_argument("--verbose",      action="store_true")
    ap.add_argument("--out-dir",      default="runs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    t_start = time.time()
    for s in args.seeds:
        r = run_one_seed(s, device=args.device,
                          protocol=args.protocol,
                          pinarx_lbfgs=args.pinarx_lbfgs,
                          narx_lbfgs=args.narx_lbfgs,
                          epochs=args.epochs,
                          verbose=args.verbose)
        results.append(r)
        # Save partial after each seed (so a crash mid-run doesn't lose progress)
        with (out_dir / "multi_seed_results.json").open("w") as f:
            json.dump(results, f, indent=2, default=str)

    summary = summarize(results)
    print(summary)
    with (out_dir / "multi_seed_summary.txt").open("w") as f:
        f.write(summary)
    print(f"\nTotal wall time: {(time.time()-t_start)/60:.1f} min")
    print(f"Saved: {out_dir / 'multi_seed_results.json'}")
    print(f"Saved: {out_dir / 'multi_seed_summary.txt'}")
