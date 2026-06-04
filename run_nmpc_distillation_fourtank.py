"""run_nmpc_distillation_fourtank.py - PINN + NMPC behavior cloning on four-tank.

Tier-1 enhancement to PINN-MPC: in addition to the physics-informed losses,
add a behavior-cloning term that matches PINN(t=1.0, state, sp, u_prev) to the
NMPC oracle's first-step action. The NMPC IS the oracle (opt_gap=0), so
direct supervision dramatically tightens the policy.

Pipeline:
  1. Sample N episodes WITH NMPC query_nmpc=True (each episode gets its u_NMPC label).
     This is expensive — NMPC is slow — but cheap relative to training.
  2. Train PINN with composite_loss_fourtank PLUS the new L_nmpc term (weight w_nmpc).
     Uses the prior LLM-best cfg as a starting point; w_nmpc is a NEW hparam.
  3. Evaluate with Bloor's optimality_gap + MAD.
  4. Optional: Two-Phase DPC refinement on top of the distilled PINN.

Usage on Colab:
  !python -u run_nmpc_distillation_fourtank.py \\
      --n-train-eps 2000 --K1 10000 --K2 10000 \\
      --w-nmpc 200.0 --output results/nmpc_distill_fourtank \\
      --run-dpc 2>&1 | tee nmpc_distill_fourtank.log

Estimated runtime on Colab T4:
  - Sample 2000 episodes WITH NMPC: ~15-25 min (slow due to IPOPT solves)
  - Train PINN (K1+K2 = 20K epochs): ~9 min
  - Eval (30 reps): ~2 min
  - Optional DPC: ~5 min
  - Total: ~30-40 min
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from agentic_pcgym.pinn_fourtank import (
    PINN_FourTank, FourTankPINNHparams, DEVICE)
from agentic_pcgym.pinn_training import train_pinn_fourtank
from agentic_pcgym.data_gen import sample_fourtank_episodes, save_episodes, load_episodes
from agentic_pcgym.evaluator import evaluate_fourtank
from agentic_pcgym.rl_refine import refine_fourtank_dpc, DPCRefineCfg


# Prior LLM-best cfg (matches FT_DEFAULT in agentic_pcgym/bench.py)
LLM_BEST_CFG = {
    "w_ode": 342.94, "w_ic": 0.30, "w_ytrk": 4.71, "w_xtrk": 0.13,
    "w_utrk": 0.04, "w_du": 1.44, "w_u": 19.25,
    "lr1": 1.67e-3, "lr2": 2.97e-4,
}


def _pinn_query_factory(net):
    @torch.no_grad()
    def pinn_query(h1, h2, h3, h4, h1_sp, h2_sp, v1_p, v2_p):
        z = lambda v: torch.tensor([float(v)], device=DEVICE)
        _, _, _, _, v1, v2 = net(
            z(1.0), z(h1), z(h2), z(h3), z(h4),
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
    ap.add_argument("--n-train-eps", type=int, default=2000,
                     help="Number of episodes to sample WITH NMPC (slow!)")
    ap.add_argument("--K1", type=int, default=10000)
    ap.add_argument("--K2", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--w-nmpc", type=float, default=200.0,
                     help="Weight on the L_nmpc behavior-cloning term")
    ap.add_argument("--output", default="results/nmpc_distill_fourtank/")
    ap.add_argument("--episodes-cache", default=None,
                     help="If set, load pre-sampled episodes (with u_nmpc) from this path")
    ap.add_argument("--n-eval-reps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    # DPC refinement (optional)
    ap.add_argument("--run-dpc", action="store_true",
                     help="After distillation, also run Two-Phase DPC refinement")
    ap.add_argument("--epochs-easy", type=int, default=50)
    ap.add_argument("--epochs-hard", type=int, default=150)
    ap.add_argument("--lr-easy", type=float, default=1e-5)
    ap.add_argument("--lr-hard", type=float, default=5e-6)
    a = ap.parse_args()

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Sample episodes with NMPC labels ----
    cache = a.episodes_cache or str(out_dir / "episodes_with_nmpc.pt")
    if Path(cache).exists():
        print(f"=== Loading cached episodes from {cache} ===")
        episodes = load_episodes(cache)
        if episodes.get("u_nmpc") is None:
            raise RuntimeError(f"Cached episodes at {cache} do NOT contain u_nmpc; "
                                "re-sample with query_nmpc=True.")
        print(f"  {len(episodes['h1_all'])} episodes, u_nmpc shape={episodes['u_nmpc'].shape}")
    else:
        print(f"=== Sampling {a.n_train_eps} four-tank episodes WITH NMPC ===")
        print("  (this is the expensive step — ~15-25 min on T4 due to IPOPT solves)")
        t0 = time.time()
        episodes = sample_fourtank_episodes(N=a.n_train_eps, seed=a.seed,
                                              query_nmpc=True, verbose=True)
        print(f"  done in {time.time()-t0:.1f}s")
        save_episodes(episodes, cache)
        print(f"  cached to {cache}\n")

    metrics_log = {}

    # --- 2. Train PINN with distillation loss ----
    print(f"\n=== Training PINN with NMPC distillation (w_nmpc={a.w_nmpc}) ===")
    hp = FourTankPINNHparams(
        K1=a.K1, K2=a.K2, bs=a.bs,
        w_nmpc=a.w_nmpc,
        **{k: float(LLM_BEST_CFG[k]) for k in
           ["w_ode", "w_ic", "w_ytrk", "w_xtrk", "w_utrk", "w_du", "w_u",
            "lr1", "lr2"]})
    print(f"  cfg: w_ode={hp.w_ode:.2f}, w_ytrk={hp.w_ytrk:.2f}, "
          f"w_du={hp.w_du:.2f}, w_nmpc={hp.w_nmpc}")
    net = PINN_FourTank().to(DEVICE)
    t0 = time.time()
    hist = train_pinn_fourtank(net, episodes, hp, verbose=True, seed=a.seed)
    print(f"\n  trained in {time.time()-t0:.1f}s")
    if hist.get("nan_at"):
        print(f"  WARNING: NaN at {hist['nan_at']}")
    torch.save(net.state_dict(), out_dir / "pinn_fourtank_distill_pretrain.pt")

    _eval(net, "DISTILLED (PINN + NMPC behavior cloning)", metrics_log,
          n_reps=a.n_eval_reps)

    # --- 3. Optional Two-Phase DPC on the distilled PINN ----
    if a.run_dpc:
        print(f"\n=== Phase 1: DPC easy-subset "
              f"({a.epochs_easy} ep, lr={a.lr_easy}) ===")
        net_p1 = copy.deepcopy(net)
        cfg_easy = DPCRefineCfg(epochs=a.epochs_easy, bs=16, lr=a.lr_easy,
                                  mode="easy", seed=a.seed)
        t0 = time.time()
        h1 = refine_fourtank_dpc(net_p1, cfg_easy, verbose=True,
                                       train_data=episodes)
        print(f"  refined in {time.time()-t0:.1f}s (ran {len(h1['loss'])} ep)")
        torch.save(net_p1.state_dict(),
                    out_dir / "pinn_fourtank_distill_p1_easy.pt")
        _eval(net_p1, "DISTILLED + Phase 1 DPC", metrics_log, n_reps=a.n_eval_reps)

        print(f"\n=== Phase 2: DPC hard-subset "
              f"({a.epochs_hard} ep, lr={a.lr_hard}) ===")
        net_p2 = copy.deepcopy(net_p1)
        cfg_hard = DPCRefineCfg(epochs=a.epochs_hard, bs=16, lr=a.lr_hard,
                                  mode="hard", seed=a.seed + 1,
                                  early_stop_patience=30)
        t0 = time.time()
        h2 = refine_fourtank_dpc(net_p2, cfg_hard, verbose=True,
                                       train_data=episodes)
        print(f"  refined in {time.time()-t0:.1f}s (ran {len(h2['loss'])} ep)")
        torch.save(net_p2.state_dict(),
                    out_dir / "pinn_fourtank_distill_p2_hard.pt")
        _eval(net_p2, "DISTILLED + Two-Phase DPC (FINAL)", metrics_log,
              n_reps=a.n_eval_reps)

    # --- 4. Summary table ----
    print("\n" + "=" * 78)
    print("=== Four-tank NMPC Distillation Results ===")
    print("=" * 78)
    print(f"{'stage':<45}{'opt gap':>14}{'MAD':>14}")
    print("-" * 73)
    for label, m in metrics_log.items():
        print(f"  {label:<43}{m['optimality_gap']:>14.4f}"
              f"{m['MAD']:>14.4f}")
    print("\nReference (from prior runs, same harness):")
    print(f"  {'LLM PINN (no distillation)':<43}{'0.5365':>14}{'0.3719':>14}")
    print(f"  {'LLM + Two-Phase DPC (no distillation)':<43}"
          f"{'0.2061':>14}{'0.1567':>14}")
    print(f"  {'SAC (our 200K-timestep baseline)':<43}{'0.0225':>14}{'0.0616':>14}")
    print(f"  {'DDPG (our 200K-timestep baseline)':<43}{'0.0031':>14}{'0.0644':>14}")

    summary = {
        "case": "fourtank_nmpc_distillation",
        "cfg": {k: float(getattr(hp, k)) for k in [
            "w_ode", "w_ic", "w_ytrk", "w_xtrk", "w_utrk", "w_du", "w_u",
            "lr1", "lr2", "w_nmpc", "K1", "K2"]},
        "n_train_eps": a.n_train_eps,
        "metrics": {k: {sk: float(sv) for sk, sv in v.items()
                          if sk in ("median_reward_pi", "median_reward_oracle",
                                     "optimality_gap", "MAD")}
                       for k, v in metrics_log.items()},
    }
    with (out_dir / "nmpc_distillation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {out_dir / 'nmpc_distillation_summary.json'}")


if __name__ == "__main__":
    main()
