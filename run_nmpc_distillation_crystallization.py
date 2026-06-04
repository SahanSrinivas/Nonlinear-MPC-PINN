"""run_nmpc_distillation_crystallization.py - NMPC behavior cloning for K2SO4 crystallization.

Mirrors run_nmpc_distillation_fourtank.py for the crystallization PC-Gym case study.
The crystallization PINN's action is a SCALAR (T_c), unlike four-tank's (v1, v2).

Pipeline:
  1. Sample N episodes WITH NMPC query_nmpc=True (records u_NMPC = T_c per ep).
  2. Train PINN with composite_loss_crystallization PLUS the L_nmpc term.
     Uses NaN-safer CRYST_DEFAULT cfg + new w_nmpc hparam.
  3. Evaluate via evaluate_crystallization (Bloor opt_gap + MAD).

Usage on Colab:
  !python -u run_nmpc_distillation_crystallization.py \\
      --n-train-eps 2000 --K1 10000 --K2 10000 \\
      --w-nmpc 200.0 \\
      --output /content/drive/MyDrive/pinn_mpc_results/nmpc_distill_cryst_seed0 \\
      --seed 0 \\
      2>&1 | tee /content/drive/MyDrive/pinn_mpc_results/distill_cryst_seed0.log

Reference numbers (Bloor 2025 Table 5, crystallization):
  DDPG = 0.0212, SAC = 0.0148, PPO = 0.0103 (best)

If our PINN+Distill lands at ~0.001-0.01, we've beaten Bloor's best RL.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from agentic_pcgym.pinn_crystallization import (
    PINN_Crystallization, CrystPINNHparams, DEVICE)
from agentic_pcgym.pinn_training import train_pinn_crystallization
from agentic_pcgym.data_gen import (
    sample_crystallization_episodes, save_episodes, load_episodes)
from agentic_pcgym.evaluator import evaluate_crystallization


# PHYSICS-INFORMED + DISTILLATION defaults (v3 architecture).
# With the bigger network ([128,128,128,128]) and sigmoid-bounded T_c output,
# we can safely re-enable physics losses at LOW weights. They act as soft
# regularizers without dominating training; L_nmpc still dominates via
# w_nmpc=500.
# T_c output is now guaranteed in [25, 50] by sigmoid → no more LSODA crashes
# from extreme T_c → physics losses can be evaluated stably.
CRYST_DEFAULT_CFG = {
    "w_ode":   0.5,     # small physics regularizer
    "w_ic":    0.1,
    "w_ytrk":  1.0,     # tracking still useful
    "w_utrk":  0.05,
    "w_du":    0.5,     # move suppression
    "w_u":     2.0,     # input bounds (sigmoid already enforces, so small)
    "w_x":     0.5,
    "lr1":     1e-3,    # bigger network → use moderate lr
    "lr2":     2e-4,
}


def _pinn_query_factory(net, t_c_min: float = 25.0, t_c_max: float = 50.0):
    """Clip Tc output to [25, 50] °C to avoid LSODA NaN on extreme values.
    Without clipping, the network can output Tc < 0 or > 100, causing the
    plant integrator to crash. With clipping, the output is always
    physically meaningful — at worst we lose some policy fidelity.
    """
    @torch.no_grad()
    def pinn_query(mu0, mu1, mu2, mu3, c, cv_sp, ln_sp):
        z = lambda v: torch.tensor([float(v)], device=DEVICE)
        Tc_ic = z(32.0)   # operating-point Tc IC
        _, _, _, _, _, Tc = net(
            z(1.0), z(mu0), z(mu1), z(mu2), z(mu3),
            z(c), z(cv_sp), z(ln_sp), Tc_ic)
        return float(max(t_c_min, min(t_c_max, Tc.item())))
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
    ap.add_argument("--n-train-eps", type=int, default=2000,
                     help="Number of episodes to sample WITH NMPC (slow!)")
    ap.add_argument("--K1", type=int, default=20000,
                     help="Phase 1 epochs (was 10000, doubled for bigger net)")
    ap.add_argument("--K2", type=int, default=20000,
                     help="Phase 2 epochs (was 10000, doubled for bigger net)")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--w-nmpc", type=float, default=500.0,
                     help="Weight on the L_nmpc behavior-cloning term. "
                          "Default 500 (vs 200 for four-tank) because "
                          "crystallization physics losses are much larger; "
                          "we need distillation to dominate.")
    ap.add_argument("--output", default="results/nmpc_distill_crystallization/")
    ap.add_argument("--episodes-cache", default=None,
                     help="If set, load pre-sampled episodes (with u_nmpc) from this path")
    ap.add_argument("--n-eval-reps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
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
        print(f"  {len(episodes['mu0_all'])} episodes, "
              f"u_nmpc shape={tuple(episodes['u_nmpc'].shape)}")
    else:
        print(f"=== Sampling {a.n_train_eps} crystallization episodes WITH NMPC ===")
        print("  (this is the expensive step — ~10-20 min on T4 due to IPOPT solves)")
        t0 = time.time()
        episodes = sample_crystallization_episodes(
            N=a.n_train_eps, seed=a.seed, query_nmpc=True, verbose=True)
        print(f"  done in {time.time()-t0:.1f}s")
        save_episodes(episodes, cache)
        print(f"  cached to {cache}\n")

    metrics_log = {}

    # --- 2. Train PINN with distillation loss ----
    print(f"\n=== Training PINN with NMPC distillation (w_nmpc={a.w_nmpc}) ===")
    hp = CrystPINNHparams(
        K1=a.K1, K2=a.K2, bs=a.bs,
        w_nmpc=a.w_nmpc,
        **{k: float(CRYST_DEFAULT_CFG[k]) for k in
           ["w_ode", "w_ic", "w_ytrk", "w_utrk", "w_du", "w_u", "w_x",
            "lr1", "lr2"]})
    print(f"  cfg: w_ode={hp.w_ode:.2f}, w_ytrk={hp.w_ytrk:.2f}, "
          f"w_du={hp.w_du:.2f}, w_nmpc={hp.w_nmpc}, "
          f"lr1={hp.lr1:.1e}, lr2={hp.lr2:.1e}")
    net = PINN_Crystallization().to(DEVICE)
    t0 = time.time()
    hist = train_pinn_crystallization(net, episodes, hp, verbose=True, seed=a.seed)
    print(f"\n  trained in {time.time()-t0:.1f}s")
    if hist.get("nan_at"):
        print(f"  WARNING: NaN at {hist['nan_at']}")
        print(f"  → distillation likely SAVED training, but check loss curves")
    torch.save(net.state_dict(),
                out_dir / "pinn_crystallization_distill_pretrain.pt")

    _eval(net, "DISTILLED Crystallization (PINN + NMPC BC)", metrics_log,
          n_reps=a.n_eval_reps)

    # --- 3. Summary table ----
    print("\n" + "=" * 78)
    print("=== Crystallization NMPC Distillation Results ===")
    print("=" * 78)
    print(f"{'stage':<55}{'opt gap':>12}{'MAD':>12}")
    print("-" * 79)
    for label, m in metrics_log.items():
        print(f"  {label:<53}{m['optimality_gap']:>12.4f}"
              f"{m['MAD']:>12.4f}")
    print("\nReference (Bloor 2025 Table 5, crystallization):")
    print(f"  {'PPO (Bloor best, 50K timesteps)':<53}{'0.0103':>12}"
          f"{'0.0013':>12}")
    print(f"  {'SAC (Bloor)':<53}{'0.0148':>12}{'0.0009':>12}")
    print(f"  {'DDPG (Bloor)':<53}{'0.0212':>12}{'0.0033':>12}")

    summary = {
        "case": "crystallization_nmpc_distillation",
        "cfg": {k: float(getattr(hp, k)) for k in [
            "w_ode", "w_ic", "w_ytrk", "w_utrk", "w_du", "w_u", "w_x",
            "lr1", "lr2", "w_nmpc", "K1", "K2"]},
        "n_train_eps": a.n_train_eps,
        "seed": a.seed,
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
