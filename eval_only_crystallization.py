"""eval_only_crystallization.py - Re-evaluate a saved crystallization PINN
without re-training. Used when the previous run trained successfully but
evaluation hung (e.g., LSODA NaN on extreme Tc outputs).

Now with Tc clipping to [25, 50] °C built in.

Usage:
  python eval_only_crystallization.py \\
      --model /content/drive/MyDrive/pinn_mpc_results/nmpc_distill_cryst_pureBC_v3/pinn_crystallization_distill_pretrain.pt \\
      --n-reps 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from agentic_pcgym.pinn_crystallization import (
    PINN_Crystallization, DEVICE)
from agentic_pcgym.evaluator import evaluate_crystallization


def pinn_query_factory(net, t_c_min=25.0, t_c_max=50.0):
    @torch.no_grad()
    def pinn_query(mu0, mu1, mu2, mu3, c, cv_sp, ln_sp):
        z = lambda v: torch.tensor([float(v)], device=DEVICE)
        Tc_ic = z(32.0)
        _, _, _, _, _, Tc = net(
            z(1.0), z(mu0), z(mu1), z(mu2), z(mu3),
            z(c), z(cv_sp), z(ln_sp), Tc_ic)
        # Critical: clip to physical Tc range to prevent LSODA NaN
        return float(max(t_c_min, min(t_c_max, Tc.item())))
    return pinn_query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                     help="Path to saved PINN .pt file")
    ap.add_argument("--n-reps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip-min", type=float, default=25.0)
    ap.add_argument("--clip-max", type=float, default=50.0)
    a = ap.parse_args()

    print(f"=== Loading PINN from {a.model} ===")
    net = PINN_Crystallization().to(DEVICE)
    net.load_state_dict(torch.load(a.model, map_location=DEVICE))
    net.eval()

    print(f"=== Evaluating on {a.n_reps} closed-loop reps "
          f"(Tc clipped to [{a.clip_min}, {a.clip_max}] °C) ===")
    q = pinn_query_factory(net, a.clip_min, a.clip_max)
    metrics = evaluate_crystallization(q, n_reps=a.n_reps, seed=a.seed,
                                          verbose=True)
    print("\n=== RESULT ===")
    print(f"  median_reward_pi:     {metrics['median_reward_pi']:>10.4f}")
    print(f"  median_reward_oracle: {metrics['median_reward_oracle']:>10.4f}")
    print(f"  optimality_gap:       {metrics['optimality_gap']:>10.4f}")
    print(f"  MAD:                  {metrics['MAD']:>10.4f}")
    print("\nReference (Bloor 2025 Table 5):")
    print(f"  DDPG: 0.0212    SAC: 0.0148    PPO: 0.0103 (best)")

    out_path = Path(a.model).parent / "eval_only_result.json"
    with open(out_path, "w") as f:
        json.dump({
            "optimality_gap": float(metrics["optimality_gap"]),
            "MAD": float(metrics["MAD"]),
            "median_reward_pi": float(metrics["median_reward_pi"]),
            "median_reward_oracle": float(metrics["median_reward_oracle"]),
            "clip_min": a.clip_min, "clip_max": a.clip_max,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
