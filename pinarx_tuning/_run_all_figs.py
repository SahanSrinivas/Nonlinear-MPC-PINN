"""Run ALL paper-figure reproductions sequentially.

Produces:
  Fig 3      = noiseless Test 1 + Test 2 (one combined image)
  Fig 7      = SNR=35 noisy Test 1 + Test 2
  Fig 8      = SNR=100 noisy Test 1 + Test 2
  Fig 5+6    = limited-data ablation (200/500/1000/2000 pts)
  Fig A.2    = training/val data layout
  Fig A.3    = validation fit

CPU-friendly: uses modest epoch counts. For paper-final figures run with
--full and a GPU.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from data_gen import (gen_grid_train_val_split, gen_test1_set, gen_test2_set,
                        gen_train_val_split,
                        add_noise, SNR_T6_NOISY_LO, SNR_T6_NOISY_HI)
from resphys_narx import (ResPhysNARXHparams, ResPhysNARXModel)
from plots import (dump_paper_figs, plot_training_data,
                     plot_fig5_style, plot_fig6_style)


def train_one(seed: int, noise: str | None, n_train_points: int | None,
                device: str, epochs: int, lbfgs: int):
    tr, va = gen_grid_train_val_split(qf_levels=10, qc_levels=10, seed=seed,
                                          n_train_points=n_train_points)
    t1, t2 = gen_test1_set(), gen_test2_set()
    if noise is not None:
        snr_vec = SNR_T6_NOISY_LO if noise == "snr35" else SNR_T6_NOISY_HI
        tr = {**tr, "y": add_noise(tr["y"], snr_vec, seed=42)}
        va = {**va, "y": add_noise(va["y"], snr_vec, seed=43)}
        t1 = {**t1, "y": add_noise(t1["y"], snr_vec, seed=44)}
        t2 = {**t2, "y": add_noise(t2["y"], snr_vec, seed=45)}
    hp = ResPhysNARXHparams(
        window=2, hidden=(200, 400, 200), activation="tanh",
        lr_adam=1e-3, lr_lbfgs=0.1,
        n_epochs_adam=epochs, lbfgs_iters=lbfgs,
        batch_size=64, weight_decay=0.0,
        early_stop_patience=120, seed=seed,
        include_u_lags=False, device=device,
        rk4_sub_steps=50, residual_l2=0.0,
    )
    t0 = time.time()
    m = ResPhysNARXModel(n_u=2, n_y=4, hp=hp)
    m.fit(tr, va, verbose=False)
    return m, tr, va, t1, t2, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                                            else "cpu")
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--out-dir", default="/tmp/all_figs")
    ap.add_argument("--full",   action="store_true",
                    help="Full epoch counts (1000/1000); default uses 200/500 for CPU.")
    args = ap.parse_args()

    epochs = 1000 if args.full else 200
    lbfgs  = 1000 if args.full else 500
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"=== Running all paper-figure reproductions ===")
    print(f"device={args.device}, epochs={epochs}, lbfgs={lbfgs}, out={out}\n")

    # ---------- Fig 3, A.2, A.3  (noiseless) ----------
    print("[1/6] noiseless (Fig 3, A.2, A.3)...")
    m_clean, tr, va, t1, t2, t_clean = train_one(args.seed, None, None,
                                                      args.device, epochs, lbfgs)
    print(f"  trained in {t_clean:.0f}s | "
            f"T1 one-step = {m_clean.eval_mae_one_step(t1):.4e}  "
            f"T2 one-step = {m_clean.eval_mae_one_step(t2):.4e}")
    paths_clean = dump_paper_figs(m_clean, tr, va, t1, t2,
                                      str(out / "fig3_A2_A3_noiseless"), "noiseless")

    # ---------- Fig 7  (SNR=35) ----------
    print("\n[2/6] SNR=35 noisy (Fig 7)...")
    m_n35, tr_n, va_n, t1_n, t2_n, t_n35 = train_one(args.seed, "snr35", None,
                                                          args.device, epochs, lbfgs)
    print(f"  trained in {t_n35:.0f}s | "
            f"T1 one-step = {m_n35.eval_mae_one_step(t1_n):.4e}  "
            f"T2 one-step = {m_n35.eval_mae_one_step(t2_n):.4e}")
    paths_n35 = dump_paper_figs(m_n35, tr_n, va_n, t1_n, t2_n,
                                    str(out / "fig7_snr35"), "snr35")

    # ---------- Fig 8  (SNR=100) ----------
    print("\n[3/6] SNR=100 noisy (Fig 8)...")
    m_n100, tr_n, va_n, t1_n, t2_n, t_n100 = train_one(args.seed, "snr100", None,
                                                            args.device, epochs, lbfgs)
    print(f"  trained in {t_n100:.0f}s | "
            f"T1 one-step = {m_n100.eval_mae_one_step(t1_n):.4e}  "
            f"T2 one-step = {m_n100.eval_mae_one_step(t2_n):.4e}")
    paths_n100 = dump_paper_figs(m_n100, tr_n, va_n, t1_n, t2_n,
                                      str(out / "fig8_snr100"), "snr100")

    # ---------- Fig 5, 6  (limited-data ablation) ----------
    print("\n[4/6] limited-data ablation (Fig 5, Fig 6)...")
    sizes = [200, 500, 1000, 2000]
    models_by_size = {}
    for n in sizes:
        print(f"  training with {n} points...")
        m, _, _, _, _, t_sz = train_one(args.seed, None, n,
                                              args.device, epochs, lbfgs)
        actual_n = min(n, 7000)   # we always allow this
        models_by_size[actual_n] = m
        print(f"    trained in {t_sz:.0f}s | "
                f"T1 = {m.eval_mae_one_step(t1):.4e}  "
                f"T2 = {m.eval_mae_one_step(t2):.4e}")
    p5a, p5b = plot_fig5_style(models_by_size, t1, t2,
                                    str(out / "fig5a_test1_CA.png"),
                                    str(out / "fig5b_test2_CA.png"))
    p6a = plot_fig6_style(models_by_size, t1,
                              str(out / "fig6a_test1_all.png"),
                              test_label="(a) Test case 1 - interpolation")
    p6b = plot_fig6_style(models_by_size, t2,
                              str(out / "fig6b_test2_all.png"),
                              test_label="(b) Test case 2 - extrapolation")

    # ---------- Fig A.2 paper-faithful (APRBS protocol) ----------
    print("\n[5/6] Fig A.2 paper-faithful (APRBS) ...")
    tr_p, va_p = gen_train_val_split(N_total=5000, N_train=2000, seed=args.seed)
    p_A2_paper = plot_training_data(tr_p, va_p,
                                          str(out / "figA2_paper_aprbs.png"))

    # ---------- Final summary ----------
    print("\n[6/6] All done.\n")
    print(f"Saved figures under  {out}\n")
    summary = [
        ("fig3_combined  (noiseless)", paths_clean["fig3"]),
        ("fig3a_test1    (noiseless)", paths_clean["test1"]),
        ("fig3b_test2    (noiseless)", paths_clean["test2"]),
        ("figA2          (dense grid)", paths_clean["training"]),
        ("figA2          (paper APRBS)", p_A2_paper),
        ("figA3          (val fit)",   paths_clean["val_fit"]),
        ("fig5a          (Test 1, C_A across sizes)", p5a),
        ("fig5b          (Test 2, C_A across sizes)", p5b),
        ("fig6a          (Test 1, all outputs across sizes)", p6a),
        ("fig6b          (Test 2, all outputs across sizes)", p6b),
        ("fig7_combined  (SNR=35)",   paths_n35["fig3"]),
        ("fig8_combined  (SNR=100)",  paths_n100["fig3"]),
    ]
    for label, path in summary:
        print(f"  {label:<48}  {path}")


if __name__ == "__main__":
    main()
