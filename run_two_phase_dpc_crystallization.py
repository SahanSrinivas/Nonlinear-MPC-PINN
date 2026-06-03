"""run_two_phase_dpc_crystallization.py - K2SO4 crystallization case study.

Mirrors run_two_phase_dpc_fourtank.py for the Bloor 2025 PC-Gym
crystallization benchmark.

Pipeline (same as fourtank):
  1. Train PINN with LLM-best cfg + optional importance weighting.
  2. Phase 1 DPC on EASY episodes (low |CV-1|+|Ln-15|/15).
  3. Phase 2 DPC on HARD episodes (extreme setpoints).
  4. Evaluate at each stage (Bloor optimality gap + MAD).
  5. Side-by-side comparison vs Bloor NMPC oracle.

Usage on Colab:
  # First run LLM-AutoOpt:
  !python -m agentic_pcgym.bench --case crystallization --tuners llm \\
      --n-trials 25 --K1 10000 --K2 10000 --bs 64 \\
      --n-train-eps 3000 --n-eval-reps 30 \\
      --out results/pcgym_crystallization

  # Then run this:
  !python run_two_phase_dpc_crystallization.py \\
      --best-cfg results/pcgym_crystallization/crystallization/llm.json

Estimated runtime on T4: ~15-20 min (after LLM-AutoOpt is done)
  - Pre-sample 3000 training episodes (no NMPC): ~30 s
  - Re-train LLM-best PINN: ~4 min
  - Phase 1 DPC (50 ep): ~3 min
  - Phase 2 DPC (150 ep): ~6 min
  - Evaluations (30 reps each, NMPC oracle slow): ~5 min total
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from agentic_pcgym.pinn_crystallization import (
    PINN_Crystallization, CrystPINNHparams, DEVICE)
from agentic_pcgym.pinn_training import train_pinn_crystallization
from agentic_pcgym.data_gen import sample_crystallization_episodes
from agentic_pcgym.evaluator import evaluate_crystallization
from agentic_pcgym.rl_refine import refine_crystallization_dpc, DPCRefineCfg


def _pinn_query_factory(net):
    @torch.no_grad()
    def pinn_query(mu0, mu1, mu2, mu3, c, cv_sp, ln_sp):
        z = lambda v: torch.tensor([float(v)], device=DEVICE)
        Tc_ic = z(32.0)   # operating-point Tc
        _, _, _, _, _, Tc = net(z(1.0), z(mu0), z(mu1), z(mu2), z(mu3),
                                    z(c), z(cv_sp), z(ln_sp), Tc_ic)
        return float(Tc.item())
    return pinn_query


def _eval(net, label, out_acc, n_reps=30):
    print(f"\n[Evaluating {label} on {n_reps} closed-loop reps]")
    q = _pinn_query_factory(net)
    metrics = evaluate_crystallization(q, n_reps=n_reps, seed=42, verbose=False)
    print(f"  median_reward_pi:     {metrics['median_reward_pi']:>10.4f}")
    print(f"  median_reward_oracle: {metrics['median_reward_oracle']:>10.4f}")
    print(f"  optimality_gap:       {metrics['optimality_gap']:>10.4f}")
    print(f"  MAD:                  {metrics['MAD']:>10.4f}")
    out_acc[label] = metrics
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-cfg",
                    default="results/pcgym_crystallization/crystallization/llm.json")
    ap.add_argument("--output",
                    default="results/two_phase_dpc_crystallization/")
    ap.add_argument("--K1", type=int, default=10000)
    ap.add_argument("--K2", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n-train-eps", type=int, default=3000)
    ap.add_argument("--n-eval-reps", type=int, default=30)
    ap.add_argument("--importance-alpha", type=float, default=0.0)
    ap.add_argument("--epochs-easy", type=int, default=50)
    ap.add_argument("--epochs-hard", type=int, default=150)
    ap.add_argument("--lr-easy", type=float, default=1e-5)
    ap.add_argument("--lr-hard", type=float, default=5e-6)
    ap.add_argument("--hard-quantile", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load LLM-best ----
    with open(a.best_cfg) as f:
        llm = json.load(f)
    best_cfg = llm["best_cfg"]
    print(f"LLM-best cfg: {best_cfg}")
    print(f"LLM-best score: {llm.get('best', 'N/A')}\n")

    # ---- Pre-sample training data ----
    print(f"=== Sampling {a.n_train_eps} crystallization episodes... ===")
    t0 = time.time()
    episodes = sample_crystallization_episodes(N=a.n_train_eps, seed=a.seed,
                                                    query_nmpc=False, verbose=True)
    print(f"  done in {time.time()-t0:.1f}s\n")

    metrics_log = {}

    # ---- Train ----
    print(f"=== Training LLM-best PINN "
          f"(importance_alpha={a.importance_alpha}) ===")
    hp = CrystPINNHparams(
        K1=a.K1, K2=a.K2, bs=a.bs,
        importance_alpha=a.importance_alpha,
        **{k: float(best_cfg[k])
           for k in ["w_ode", "w_ic", "w_ytrk", "w_utrk",
                     "w_du", "w_u", "w_x", "lr1", "lr2"]})
    net = PINN_Crystallization().to(DEVICE)
    t0 = time.time()
    hist = train_pinn_crystallization(net, episodes, hp, verbose=False,
                                          seed=a.seed)
    print(f"  trained in {time.time()-t0:.1f}s")
    if hist.get("nan_at"):
        print(f"  WARNING: NaN at {hist['nan_at']}")
    torch.save(net.state_dict(), out_dir / "pinn_cryst_llm_pretrain.pt")

    m_base = _eval(net, "BASELINE (LLM only)", metrics_log,
                     n_reps=a.n_eval_reps)

    # ---- Phase 1: easy-subset DPC ----
    print(f"\n=== Phase 1: DPC easy-subset "
          f"({a.epochs_easy} ep, lr={a.lr_easy}) ===")
    net_p1 = copy.deepcopy(net)
    cfg_easy = DPCRefineCfg(epochs=a.epochs_easy, bs=16, lr=a.lr_easy,
                              mode="easy", hard_quantile=a.hard_quantile,
                              seed=a.seed)
    t0 = time.time()
    h1 = refine_crystallization_dpc(net_p1, cfg_easy, verbose=True,
                                          train_data=episodes)
    print(f"  refined in {time.time()-t0:.1f}s (ran {len(h1['loss'])} ep)")
    torch.save(net_p1.state_dict(), out_dir / "pinn_cryst_p1_easy.pt")
    m_p1 = _eval(net_p1, "AFTER Phase 1 (easy DPC)", metrics_log,
                   n_reps=a.n_eval_reps)

    # ---- Phase 2: hard-subset DPC ----
    print(f"\n=== Phase 2: DPC hard-subset "
          f"({a.epochs_hard} ep, lr={a.lr_hard}) ===")
    net_p2 = copy.deepcopy(net_p1)
    cfg_hard = DPCRefineCfg(epochs=a.epochs_hard, bs=16, lr=a.lr_hard,
                              mode="hard", hard_quantile=a.hard_quantile,
                              seed=a.seed + 1, early_stop_patience=30)
    t0 = time.time()
    h2 = refine_crystallization_dpc(net_p2, cfg_hard, verbose=True,
                                          train_data=episodes)
    print(f"  refined in {time.time()-t0:.1f}s (ran {len(h2['loss'])} ep)")
    torch.save(net_p2.state_dict(), out_dir / "pinn_cryst_p2_hard.pt")
    m_p2 = _eval(net_p2, "AFTER Phase 2 (hard DPC) - FINAL", metrics_log,
                   n_reps=a.n_eval_reps)

    # ---- Side-by-side ----
    print("\n" + "=" * 78)
    print("=== Crystallization: Two-Phase DPC RESULTS vs NMPC (Bloor 2025) ===")
    print("=" * 78)
    rows = [
        ("median_reward_pi (PINN)",
         f"{m_base['median_reward_pi']:.4f}",
         f"{m_p1['median_reward_pi']:.4f}",
         f"{m_p2['median_reward_pi']:.4f}",
         f"{m_base['median_reward_oracle']:.4f} (oracle)"),
        ("optimality_gap",
         f"{m_base['optimality_gap']:.4f}",
         f"{m_p1['optimality_gap']:.4f}",
         f"{m_p2['optimality_gap']:.4f}",
         "0.0 (oracle)"),
        ("MAD",
         f"{m_base['MAD']:.4f}",
         f"{m_p1['MAD']:.4f}",
         f"{m_p2['MAD']:.4f}",
         "-"),
    ]
    print(f"{'metric':<28}{'LLM':>14}{'+P1 easy':>14}{'+P2 hard':>14}"
          f"{'NMPC':>22}")
    print("-" * 92)
    for r in rows:
        print(f"  {r[0]:<28}{r[1]:>14}{r[2]:>14}{r[3]:>14}{r[4]:>22}")

    candidates = {"LLM only": m_base["optimality_gap"],
                   "+P1 easy":  m_p1["optimality_gap"],
                   "+P2 hard":  m_p2["optimality_gap"]}
    best_label = min(candidates, key=candidates.get)
    best_gap = candidates[best_label]
    print()
    print(f"  >>> Best stage: {best_label} (optimality gap = {best_gap:.4f}) <<<")

    summary = {
        "case": "crystallization",
        "llm_best_cfg": best_cfg,
        "metrics": {k: {sk: float(sv) if isinstance(sv, (int, float, np.floating))
                          else sv for sk, sv in v.items()
                          if sk in ("median_reward_pi", "median_reward_oracle",
                                     "optimality_gap", "MAD")}
                       for k, v in metrics_log.items()},
        "headline": {"best_stage": best_label, "best_gap": float(best_gap)},
        "config": {
            "K1": a.K1, "K2": a.K2, "bs": a.bs,
            "n_train_eps": a.n_train_eps,
            "importance_alpha": a.importance_alpha,
            "epochs_easy": a.epochs_easy, "epochs_hard": a.epochs_hard,
            "lr_easy": a.lr_easy, "lr_hard": a.lr_hard,
            "hard_quantile": a.hard_quantile,
            "n_eval_reps": a.n_eval_reps,
        },
    }
    with (out_dir / "two_phase_dpc_crystallization.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {out_dir / 'two_phase_dpc_crystallization.json'}")
    print(f"Models saved: pinn_cryst_llm_pretrain.pt, "
          f"pinn_cryst_p1_easy.pt, pinn_cryst_p2_hard.pt")


if __name__ == "__main__":
    main()
