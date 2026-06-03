"""run_two_phase_dpc_fourtank.py - Four-tank MIMO case study.

Applies the SISO methodology (LLM-AutoOpt + two-phase DPC + Kardamaki-style
training-data refinement) to the Bloor 2025 PC-Gym four-tank benchmark.

Pipeline:
  1. Run LLM-AutoOpt for hparam search (25 trials, K=10000) - OR load saved cfg.
  2. Train PINN with LLM-best + optional importance weighting.
  3. Phase 1 DPC (easy episodes, lr=1e-5).
  4. Phase 2 DPC (hard episodes, lr=5e-6).
  5. Evaluate at each stage (Bloor optimality gap + MAD).
  6. Side-by-side comparison vs the Bloor NMPC oracle baseline (oracle reward
     itself, since Bloor doesn't publish a PINN baseline for four-tank).

Usage on Colab:
  # First run LLM-AutoOpt:
  !python -m agentic_pcgym.bench --case fourtank --tuners llm \\
      --n-trials 25 --K1 10000 --K2 10000 --bs 100 \\
      --n-train-eps 5000 --n-eval-reps 30 \\
      --out results/pcgym_fourtank

  # Then run this two-phase DPC on the LLM-best cfg:
  !python run_two_phase_dpc_fourtank.py \\
      --best-cfg results/pcgym_fourtank/fourtank/llm.json

Estimated runtime on T4: ~10-15 min (after LLM-AutoOpt is done)
  - Pre-sample 5000 training episodes (with NMPC queries): ~3 min
  - Re-train LLM-best PINN: ~3-4 min
  - Phase 1 DPC (50 ep): ~2 min
  - Phase 2 DPC (150 ep): ~5 min
  - Evaluations (50 reps each, NMPC oracle is slow): ~3 min total
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from agentic_pcgym.pinn_fourtank import (PINN_FourTank, FourTankPINNHparams,
                                            DEVICE)
from agentic_pcgym.pinn_training import train_pinn_fourtank
from agentic_pcgym.data_gen import sample_fourtank_episodes
from agentic_pcgym.evaluator import evaluate_fourtank
from agentic_pcgym.rl_refine import refine_fourtank_dpc, DPCRefineCfg
from agentic_pcgym.nmpc_fourtank import FourTankNMPC


def _pinn_query_factory(net):
    @torch.no_grad()
    def pinn_query(h1, h2, h3, h4, h1_sp, h2_sp, v1_p, v2_p):
        z = lambda v: torch.tensor([float(v)], device=DEVICE)
        _, _, _, _, v1, v2 = net(z(1.0), z(h1), z(h2), z(h3), z(h4),
                                    z(h1_sp), z(h2_sp), z(v1_p), z(v2_p))
        return (float(v1.item()), float(v2.item()))
    return pinn_query


def _eval(net, label, out_acc, n_reps=30):
    print(f"\n[Evaluating {label} on {n_reps} closed-loop reps]")
    q = _pinn_query_factory(net)
    metrics = evaluate_fourtank(q, n_reps=n_reps, seed=42, verbose=False)
    print(f"  median_reward_pi:     {metrics['median_reward_pi']:>10.4f}")
    print(f"  median_reward_oracle: {metrics['median_reward_oracle']:>10.4f}")
    print(f"  optimality_gap:       {metrics['optimality_gap']:>10.4f}")
    print(f"  MAD:                  {metrics['MAD']:>10.4f}")
    out_acc[label] = metrics
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-cfg",
                    default="results/pcgym_fourtank/fourtank/llm.json",
                    help="LLM-best cfg JSON (from agentic_pcgym.bench LLM run)")
    ap.add_argument("--output", default="results/two_phase_dpc_fourtank/")
    ap.add_argument("--K1", type=int, default=10000)
    ap.add_argument("--K2", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n-train-eps", type=int, default=5000,
                    help="Pre-sampled training episodes")
    ap.add_argument("--n-eval-reps", type=int, default=30,
                    help="Closed-loop evaluation reps per stage")
    ap.add_argument("--importance-alpha", type=float, default=0.0,
                    help="Hard-episode oversample during PINN training "
                         "(0 = uniform, 1 = up to 2x weight)")
    ap.add_argument("--epochs-easy", type=int, default=50,
                    help="DPC Phase 1 epochs (easy subset)")
    ap.add_argument("--epochs-hard", type=int, default=150,
                    help="DPC Phase 2 epochs (hard subset)")
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
    print(f"LLM-best score (negative median reward): "
          f"{llm.get('best', 'N/A')}\n")

    # ---- Pre-sample training data (used for BOTH training + DPC) ----
    print(f"=== Sampling {a.n_train_eps} fourtank training episodes... ===")
    t0 = time.time()
    episodes = sample_fourtank_episodes(N=a.n_train_eps, seed=a.seed,
                                            query_nmpc=False, verbose=True)
    print(f"  done in {time.time()-t0:.1f}s\n")

    metrics_log = {}

    # ---- Train PINN with LLM-best + importance weighting ----
    print(f"=== Training LLM-best PINN "
          f"(importance_alpha={a.importance_alpha}) ===")
    hp = FourTankPINNHparams(
        K1=a.K1, K2=a.K2, bs=a.bs,
        importance_alpha=a.importance_alpha,
        **{k: float(best_cfg[k])
           for k in ["w_ode", "w_ic", "w_ytrk", "w_xtrk", "w_utrk",
                     "w_du", "w_u", "lr1", "lr2"]})
    net = PINN_FourTank().to(DEVICE)
    t0 = time.time()
    hist = train_pinn_fourtank(net, episodes, hp, verbose=False,
                                  seed=a.seed)
    print(f"  trained in {time.time()-t0:.1f}s")
    if hist.get("nan_at"):
        print(f"  WARNING: NaN during training at {hist['nan_at']}")
    torch.save(net.state_dict(), out_dir / "pinn_fourtank_llm_pretrain.pt")

    # ---- Baseline eval ----
    m_base = _eval(net, "BASELINE (LLM only)", metrics_log,
                     n_reps=a.n_eval_reps)

    # ---- Phase 1: easy-subset DPC ----
    print(f"\n=== Phase 1: DPC easy-subset "
          f"({a.epochs_easy} ep, lr={a.lr_easy}) ===")
    net_p1 = copy.deepcopy(net)
    cfg_easy = DPCRefineCfg(epochs=a.epochs_easy, bs=32, lr=a.lr_easy,
                              mode="easy", hard_quantile=a.hard_quantile,
                              seed=a.seed)
    t0 = time.time()
    h1 = refine_fourtank_dpc(net_p1, cfg_easy, verbose=True,
                                train_data=episodes)
    print(f"  refined in {time.time()-t0:.1f}s "
          f"(ran {len(h1['loss'])} ep)")
    torch.save(net_p1.state_dict(), out_dir / "pinn_fourtank_p1_easy.pt")
    m_p1 = _eval(net_p1, "AFTER Phase 1 (easy DPC)", metrics_log,
                   n_reps=a.n_eval_reps)

    # ---- Phase 2: hard-subset DPC on top of P1 ----
    print(f"\n=== Phase 2: DPC hard-subset "
          f"({a.epochs_hard} ep, lr={a.lr_hard}) ===")
    net_p2 = copy.deepcopy(net_p1)
    cfg_hard = DPCRefineCfg(epochs=a.epochs_hard, bs=32, lr=a.lr_hard,
                              mode="hard", hard_quantile=a.hard_quantile,
                              seed=a.seed + 1, early_stop_patience=30)
    t0 = time.time()
    h2 = refine_fourtank_dpc(net_p2, cfg_hard, verbose=True,
                                train_data=episodes)
    print(f"  refined in {time.time()-t0:.1f}s "
          f"(ran {len(h2['loss'])} ep)")
    torch.save(net_p2.state_dict(), out_dir / "pinn_fourtank_p2_hard.pt")
    m_p2 = _eval(net_p2, "AFTER Phase 2 (hard DPC) - FINAL", metrics_log,
                   n_reps=a.n_eval_reps)

    # ---- Side-by-side ----
    print("\n" + "=" * 78)
    print("=== Four-tank: Two-Phase DPC RESULTS vs NMPC oracle (Bloor 2025) ===")
    print("=" * 78)
    headline_metric = "median_reward_pi"   # higher = better
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

    # Headline: which stage has BEST median reward (lowest optimality gap)?
    candidates = {"LLM only": m_base["optimality_gap"],
                   "+P1 easy":  m_p1["optimality_gap"],
                   "+P2 hard":  m_p2["optimality_gap"]}
    best_label = min(candidates, key=candidates.get)
    best_gap = candidates[best_label]
    print()
    print(f"  >>> Best stage: {best_label} (optimality gap = {best_gap:.4f}) <<<")
    print(f"  Lower gap = closer to NMPC oracle. Gap=0 means matching oracle.")

    # ---- Save summary ----
    summary = {
        "case": "fourtank",
        "llm_best_cfg": best_cfg,
        "metrics": {k: {sk: float(sv) if isinstance(sv, (int, float, np.floating))
                          else sv for sk, sv in v.items()
                          if sk in ("median_reward_pi",
                                     "median_reward_oracle",
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
    with (out_dir / "two_phase_dpc_fourtank.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {out_dir / 'two_phase_dpc_fourtank.json'}")
    print(f"Models saved: pinn_fourtank_llm_pretrain.pt, "
          f"pinn_fourtank_p1_easy.pt, pinn_fourtank_p2_hard.pt")


if __name__ == "__main__":
    main()
