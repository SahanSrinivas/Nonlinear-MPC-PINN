"""Phase 1: Two-phase DPC + importance weighting on the current LLM-best config.

Run after STEP 2 of runpod_setup.sh has produced
results/llm_phaseA/llm_trials.json.

Steps:
  A. Train PINN with LLM-best hparams + importance_d0_alpha=1.0
     (disturbance episodes weighted 2x during training).
  B. Evaluate baseline on Kardamaki samples.
  C. DPC refine on TRACKING ONLY (50 epochs at lr=1e-5).
  D. Evaluate after tracking refine.
  E. DPC refine on DISTURBANCE ONLY (200 epochs at lr=5e-6,
     disturbance_oversample=1.5).
  F. Evaluate after disturbance refine.
  G. Print Kardamaki side-by-side comparison.
"""
from __future__ import annotations

import json
import time
import torch

from agentic_pinn_mpc.bench import load_training_data
from agentic_pinn_mpc.pinn_siso import PINNHparams, train_pinn_siso
from agentic_pinn_mpc.evaluate import evaluate_model
from agentic_pinn_mpc.rl_refine import refine_with_dpc, DPCRefineCfg


KARD = {
    "tracking_mean_offset_m":   0.0161,
    "tracking_max_offset_m":    0.0322,
    "disturbance_mean_offset_m":0.0129,
    "disturbance_max_offset_m": 0.0453,
}
KARD_combined = (0.5*(KARD["tracking_mean_offset_m"]
                     + KARD["disturbance_mean_offset_m"])
                  + 0.25*(KARD["tracking_max_offset_m"]
                          + KARD["disturbance_max_offset_m"]))


def _print_metrics(name, m):
    print(f"\n{name}:")
    for k in ["tracking_mean_offset_m", "tracking_max_offset_m",
              "disturbance_mean_offset_m", "disturbance_max_offset_m",
              "combined_score"]:
        print(f"  {k}: {m[k]:.4f}")


def main():
    # ---- Load LLM-best ----
    with open("results/llm_phaseA/llm_trials.json") as f:
        llm = json.load(f)
    best_cfg = llm["best_cfg"]
    print(f"LLM-best cfg: {best_cfg}")
    print(f"LLM-best score (Step 2): {llm['best_score']:.4f}")

    # ---- A. Train with importance weighting ----
    print("\n=== Phase 1A: train PINN with importance_d0_alpha=1.0 ===")
    hp = PINNHparams(
        **{k: v for k, v in best_cfg.items() if k in PINNHparams.__dataclass_fields__},
        K1=10000, K2=10000, bs=100,
        importance_d0_alpha=1.0,  # disturbance episodes weighted up to 2x
    )
    x0_all, u0_all, ysp_all, d0_all = load_training_data()
    t0 = time.time()
    model, _ = train_pinn_siso(hp, x0_all, u0_all, ysp_all, d0_all,
                                 verbose=False)
    print(f"  Train time: {time.time()-t0:.1f}s")

    # ---- B. Baseline eval ----
    m_baseline = evaluate_model(model, n_tracking=500, n_disturbance=500,
                                  use_kardamaki_samples=True)
    _print_metrics("After importance-weighted training (baseline)", m_baseline)

    # ---- C. DPC refine on tracking only ----
    print("\n=== Phase 1B: DPC refine TRACKING-ONLY (50 ep, lr=1e-5) ===")
    cfg_track = DPCRefineCfg(epochs=50, bs=32, lr=1e-5, mode="tracking")
    t0 = time.time()
    model, _ = refine_with_dpc(model, cfg_track, verbose=True)
    print(f"  Refine time: {time.time()-t0:.1f}s")

    m_after_track = evaluate_model(model, n_tracking=500, n_disturbance=500,
                                      use_kardamaki_samples=True)
    _print_metrics("After tracking-only DPC", m_after_track)

    # ---- D. DPC refine on disturbance only ----
    print("\n=== Phase 1C: DPC refine DISTURBANCE-ONLY (200 ep, lr=5e-6, oversample=1.5) ===")
    cfg_dist = DPCRefineCfg(epochs=200, bs=32, lr=5e-6, mode="disturbance",
                              disturbance_oversample=1.5)
    t0 = time.time()
    model, _ = refine_with_dpc(model, cfg_dist, verbose=True)
    print(f"  Refine time: {time.time()-t0:.1f}s")

    m_final = evaluate_model(model, n_tracking=500, n_disturbance=500,
                                use_kardamaki_samples=True)
    _print_metrics("After disturbance-only DPC (FINAL)", m_final)

    # ---- E. Side-by-side vs Kardamaki ----
    print("\n" + "=" * 70)
    print("=== Phase 1 RESULTS: side-by-side vs Kardamaki 2026 Table 4 ===")
    print("=" * 70)
    print(f"{'metric':<32}{'Kardamaki':>12}{'baseline':>12}"
          f"{'+track DPC':>12}{'+dist DPC':>12}{'vs Kard':>10}")
    print("-" * 90)
    for k in ["tracking_mean_offset_m", "tracking_max_offset_m",
              "disturbance_mean_offset_m", "disturbance_max_offset_m"]:
        kv = KARD[k]
        b  = m_baseline[k]
        t  = m_after_track[k]
        f  = m_final[k]
        print(f"  {k:<30}{kv:>12.4f}{b:>12.4f}"
              f"{t:>12.4f}{f:>12.4f}{f/kv:>9.2f}x")
    print(f"  {'combined_score':<30}{KARD_combined:>12.4f}"
          f"{m_baseline['combined_score']:>12.4f}"
          f"{m_after_track['combined_score']:>12.4f}"
          f"{m_final['combined_score']:>12.4f}"
          f"{m_final['combined_score']/KARD_combined:>9.2f}x")

    if m_final["combined_score"] < KARD_combined:
        delta = (KARD_combined - m_final["combined_score"]) / KARD_combined * 100
        print(f"\n  >>> Phase 1 (two-phase DPC + importance) BEATS "
              f"Kardamaki by {delta:.1f}%  <<<")
    else:
        delta = (m_final["combined_score"] - KARD_combined) / KARD_combined * 100
        print(f"\n  >>> Phase 1 is {delta:.1f}% above Kardamaki  <<<")

    # ---- F. Save ----
    with open("results/phase1_two_phase_dpc.json", "w") as f:
        json.dump({
            "llm_best_cfg": best_cfg,
            "metrics_baseline": m_baseline,
            "metrics_after_tracking_dpc": m_after_track,
            "metrics_after_disturbance_dpc": m_final,
            "kardamaki": KARD,
            "kardamaki_combined": KARD_combined,
        }, f, indent=2)
    print("\nResults saved to results/phase1_two_phase_dpc.json")


if __name__ == "__main__":
    main()
