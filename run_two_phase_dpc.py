"""run_two_phase_dpc.py - Apply new two-phase DPC on LLM-best PINN.

Differences vs original STEP 3:
  1. DPC samples from Kardamaki training data (NOT random sample_episodes).
     This fixes the distribution-shift bug that wrecked the OLD DPC.
  2. Two-phase: tracking-only refinement → disturbance-only refinement.
  3. Lower learning rates (1e-5 then 5e-6) to prevent overshoot.
  4. Optional importance weighting during initial train.

Assumes you've already pushed the latest code and pulled on Colab.
LLM-best config is read from results/llm_phaseA/llm_trials.json.

Usage on Colab:
  !cd /content/Nonlinear-MPC-PINN && git pull
  !python run_two_phase_dpc.py

Expected runtime on T4: ~6-8 minutes
  - Re-train LLM-best PINN: ~4 min
  - DPC tracking phase (50 ep): ~30 s
  - DPC disturbance phase (150 ep): ~90 s
  - Evaluations + saves: ~30 s
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from agentic_pinn_mpc.bench import load_training_data
from agentic_pinn_mpc.pinn_siso import (DEVICE, PINNHparams, PINN_Controller,
                                          train_pinn_siso)
from agentic_pinn_mpc.evaluate import evaluate_model
from agentic_pinn_mpc.rl_refine import refine_with_dpc, DPCRefineCfg


KARD = {
    "tracking_mean_offset_m":   0.0161,
    "tracking_max_offset_m":    0.0322,
    "disturbance_mean_offset_m":0.0129,
    "disturbance_max_offset_m": 0.0453,
}
KARD_combined = (0.5 * (KARD["tracking_mean_offset_m"]
                         + KARD["disturbance_mean_offset_m"])
                  + 0.25 * (KARD["tracking_max_offset_m"]
                            + KARD["disturbance_max_offset_m"]))


def _eval(model, label, out_acc=None):
    m = evaluate_model(model, n_tracking=500, n_disturbance=500,
                          use_kardamaki_samples=True)
    print(f"\n[{label}]")
    for k in ["tracking_mean_offset_m", "tracking_max_offset_m",
              "disturbance_mean_offset_m", "disturbance_max_offset_m",
              "combined_score"]:
        print(f"  {k}: {m[k]:.4f}")
    if out_acc is not None:
        out_acc[label] = m
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-cfg",
                    default="results/llm_phaseA/llm_trials.json")
    ap.add_argument("--saved-model", default="",
                    help="Optional: skip retraining, load this .pt instead")
    ap.add_argument("--output", default="results/two_phase_dpc/")
    ap.add_argument("--K1", type=int, default=10000)
    ap.add_argument("--K2", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--importance-alpha", type=float, default=0.0,
                    help="Disturbance-episode oversampling during training "
                         "(0.0 = uniform Kardamaki, 1.0 = up to 2x for d0=0.4)")
    ap.add_argument("--epochs-tracking", type=int, default=50,
                    help="DPC epochs for tracking-only phase")
    ap.add_argument("--epochs-disturbance", type=int, default=150,
                    help="DPC epochs for disturbance-only phase")
    ap.add_argument("--lr-tracking", type=float, default=1e-5)
    ap.add_argument("--lr-disturbance", type=float, default=5e-6)
    ap.add_argument("--oversample-dist", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load LLM-best ----
    with open(a.best_cfg) as f:
        llm = json.load(f)
    best_cfg = llm["best_cfg"]
    print(f"LLM-best cfg: {best_cfg}")
    print(f"LLM-best score (from Step 2): {llm.get('best_score', 'N/A')}")

    # ---- Load training data (used for BOTH training + DPC) ----
    print("\nLoading Kardamaki training data...")
    x0_all, u0_all, ysp_all, d0_all = load_training_data()
    print(f"  {x0_all.shape[0]} episodes "
          f"(d0=0: ~{(d0_all == 0).sum().item()}, "
          f"d0>0: ~{(d0_all > 0).sum().item()})")

    metrics_log = {}

    # ---- Train (or load) LLM-best PINN ----
    if a.saved_model and Path(a.saved_model).exists():
        print(f"\nLoading saved model: {a.saved_model}")
        model = PINN_Controller().to(DEVICE)
        model.load_state_dict(torch.load(a.saved_model, map_location=DEVICE))
    else:
        print(f"\n=== Training LLM-best PINN (importance_alpha={a.importance_alpha}) ===")
        hp = PINNHparams(
            **{k: v for k, v in best_cfg.items()
               if k in PINNHparams.__dataclass_fields__},
            K1=a.K1, K2=a.K2, bs=a.bs,
            importance_d0_alpha=a.importance_alpha,
        )
        t0 = time.time()
        model, _ = train_pinn_siso(hp, x0_all, u0_all, ysp_all, d0_all,
                                      verbose=False, seed=a.seed)
        print(f"  trained in {time.time()-t0:.1f}s")
        torch.save(model.state_dict(), out_dir / "pinn_llm_pretrain.pt")

    # ---- Baseline eval ----
    m_base = _eval(model, "BASELINE (LLM only, no DPC)", metrics_log)

    # ---- Save a baseline copy before we mutate ----
    model_track = copy.deepcopy(model)

    # ---- Phase 1: tracking-only DPC ----
    print(f"\n=== Phase 1: DPC tracking-only "
          f"({a.epochs_tracking} ep, lr={a.lr_tracking}) ===")
    cfg_track = DPCRefineCfg(epochs=a.epochs_tracking, bs=32,
                                lr=a.lr_tracking, mode="tracking",
                                seed=a.seed, early_stop_patience=20)
    t0 = time.time()
    model_track, hist_t = refine_with_dpc(
        model_track, cfg_track, verbose=True,
        train_data=(x0_all, u0_all, ysp_all, d0_all))
    print(f"  refined in {time.time()-t0:.1f}s "
          f"(ran {len(hist_t['loss'])} ep)")
    m_track = _eval(model_track, "AFTER Phase 1 (tracking-only DPC)",
                      metrics_log)

    # ---- Phase 2: disturbance-only DPC ----
    model_dpc = copy.deepcopy(model_track)
    print(f"\n=== Phase 2: DPC disturbance-only "
          f"({a.epochs_disturbance} ep, lr={a.lr_disturbance}, "
          f"oversample={a.oversample_dist}) ===")
    cfg_dist = DPCRefineCfg(epochs=a.epochs_disturbance, bs=32,
                              lr=a.lr_disturbance, mode="disturbance",
                              disturbance_oversample=a.oversample_dist,
                              seed=a.seed + 1, early_stop_patience=30)
    t0 = time.time()
    model_dpc, hist_d = refine_with_dpc(
        model_dpc, cfg_dist, verbose=True,
        train_data=(x0_all, u0_all, ysp_all, d0_all))
    print(f"  refined in {time.time()-t0:.1f}s "
          f"(ran {len(hist_d['loss'])} ep)")
    m_final = _eval(model_dpc, "AFTER Phase 2 (disturbance-only DPC) - FINAL",
                       metrics_log)

    # ---- Save models + history ----
    torch.save(model_track.state_dict(), out_dir / "pinn_llm_track.pt")
    torch.save(model_dpc.state_dict(), out_dir / "pinn_llm_dpc.pt")

    # ---- Side-by-side comparison ----
    print("\n" + "=" * 78)
    print("=== RESULTS: Two-Phase DPC (Kardamaki-distribution) vs Kardamaki ===")
    print("=" * 78)
    print(f"{'metric':<32}{'Kardamaki':>11}{'LLM':>11}"
          f"{'+P1 track':>12}{'+P2 dist':>11}{'vs Kard':>10}")
    print("-" * 87)
    for k in ["tracking_mean_offset_m", "tracking_max_offset_m",
              "disturbance_mean_offset_m", "disturbance_max_offset_m"]:
        kv = KARD[k]
        b  = m_base[k]
        t  = m_track[k]
        f  = m_final[k]
        print(f"  {k:<30}{kv:>11.4f}{b:>11.4f}"
              f"{t:>12.4f}{f:>11.4f}{f/kv:>9.2f}x")
    print(f"  {'combined_score':<30}{KARD_combined:>11.4f}"
          f"{m_base['combined_score']:>11.4f}"
          f"{m_track['combined_score']:>12.4f}"
          f"{m_final['combined_score']:>11.4f}"
          f"{m_final['combined_score']/KARD_combined:>9.2f}x")

    headline_score = min(m_base["combined_score"],
                          m_track["combined_score"],
                          m_final["combined_score"])
    headline_label = ["LLM only", "+P1 track", "+P2 dist"][
        [m_base["combined_score"], m_track["combined_score"],
         m_final["combined_score"]].index(headline_score)]
    print()
    if headline_score < KARD_combined:
        delta = (KARD_combined - headline_score) / KARD_combined * 100
        print(f"  >>> Best ({headline_label}) BEATS Kardamaki "
              f"by {delta:.1f}%  <<<")
    else:
        delta = (headline_score - KARD_combined) / KARD_combined * 100
        print(f"  >>> Best ({headline_label}) is {delta:.1f}% "
              f"above Kardamaki  <<<")

    # ---- Save summary ----
    summary = {
        "llm_best_cfg": best_cfg,
        "kardamaki_reference": {**KARD, "combined": KARD_combined},
        "metrics": {
            "baseline_LLM_only": m_base,
            "after_phase1_tracking": m_track,
            "after_phase2_disturbance": m_final,
        },
        "headline": {
            "label": headline_label,
            "score": headline_score,
            "vs_kardamaki_pct": (headline_score - KARD_combined) /
                                  KARD_combined * 100,
        },
        "config": {
            "K1": a.K1, "K2": a.K2, "bs": a.bs,
            "importance_alpha": a.importance_alpha,
            "epochs_tracking": a.epochs_tracking,
            "epochs_disturbance": a.epochs_disturbance,
            "lr_tracking": a.lr_tracking,
            "lr_disturbance": a.lr_disturbance,
            "oversample_dist": a.oversample_dist,
        },
    }
    with open(out_dir / "two_phase_dpc_result.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {out_dir / 'two_phase_dpc_result.json'}")
    print(f"Models saved: pinn_llm_pretrain.pt, pinn_llm_track.pt, "
          f"pinn_llm_dpc.pt")


if __name__ == "__main__":
    main()
