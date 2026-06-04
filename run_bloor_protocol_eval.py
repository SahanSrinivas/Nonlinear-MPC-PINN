"""run_bloor_protocol_eval.py - Re-evaluate trained models under Bloor Fig 7
step-change setpoint protocol (matches Bloor 2025 Table 5 setup).

Loads previously-trained models from disk and evaluates them under BOTH:
  - Original protocol: random constant setpoint per episode (broader range)
  - Bloor protocol:    Fig 7-style step changes within each episode

This gives an apples-to-apples comparison with Bloor Table 5 numbers without
re-training anything (just re-runs evaluate_fourtank with step_setpoints=True).

Usage on Colab:
  !python -u run_bloor_protocol_eval.py \\
      --pinn-distill results/nmpc_distill_fourtank/pinn_fourtank_distill_pretrain.pt \\
      --pinn-llm-dpc results/two_phase_dpc_fourtank/pinn_fourtank_p2_hard.pt \\
      --sac results/rl_baselines_fourtank/sac_fourtank.zip \\
      --ddpg results/rl_baselines_fourtank/ddpg_fourtank.zip \\
      --n-reps 30

All --* model args are OPTIONAL; missing models are skipped.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from agentic_pcgym.pinn_fourtank import PINN_FourTank, DEVICE
from agentic_pcgym.evaluator import evaluate_fourtank


def _pinn_query(net):
    @torch.no_grad()
    def query(h1, h2, h3, h4, h1_sp, h2_sp, v1_p, v2_p):
        z = lambda v: torch.tensor([float(v)], device=DEVICE)
        _, _, _, _, v1, v2 = net(
            z(1.0), z(h1), z(h2), z(h3), z(h4),
            z(h1_sp), z(h2_sp), z(v1_p), z(v2_p))
        return (float(v1.item()), float(v2.item()))
    return query


def _load_pinn(path: str | Path):
    net = PINN_FourTank().to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net


def _load_sb3(path: str | Path, algo: str):
    """Load a stable_baselines3 model (SAC/DDPG/PPO)."""
    if algo == "SAC":
        from stable_baselines3 import SAC
        return SAC.load(path)
    if algo == "DDPG":
        from stable_baselines3 import DDPG
        return DDPG.load(path)
    if algo == "PPO":
        from stable_baselines3 import PPO
        return PPO.load(path)
    raise ValueError(f"Unknown algo: {algo}")


def _sb3_query(model):
    """Wrap a stable_baselines3 model as a controller_query function."""
    def query(h1, h2, h3, h4, h1_sp, h2_sp, v1_p, v2_p):
        obs = np.array([h1, h2, h3, h4, h1_sp, h2_sp, v1_p, v2_p],
                        dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        return (float(action[0]), float(action[1]))
    return query


def _eval_both(label: str, query, n_reps: int, results: dict):
    print(f"\n=== {label} ===")
    for protocol, step_flag in [("constant setpoints", False),
                                  ("Bloor step changes", True)]:
        print(f"  Evaluating protocol: {protocol}...")
        t0 = time.time()
        m = evaluate_fourtank(query, n_reps=n_reps, seed=42, verbose=False,
                               step_setpoints=step_flag)
        elapsed = time.time() - t0
        n_e = m["n_setpoint_errors"]
        print(f"    median_reward_pi:     {m['median_reward_pi']:>10.4f}")
        print(f"    median_reward_oracle: {m['median_reward_oracle']:>10.4f}")
        print(f"    optimality_gap:       {m['optimality_gap']:>10.4f}"
              f"  (N_e={n_e}, raw_gap={m['optimality_gap']*n_e:.4f})")
        print(f"    MAD:                  {m['MAD']:>10.4f}")
        print(f"    eval time: {elapsed:.1f}s")
        results.setdefault(label, {})[protocol] = {
            "opt_gap": m["optimality_gap"],
            "MAD": m["MAD"],
            "median_reward_pi": m["median_reward_pi"],
            "median_reward_oracle": m["median_reward_oracle"],
            "n_setpoint_errors": n_e,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinn-distill", default=None,
                     help="Path to NMPC-distilled PINN .pt")
    ap.add_argument("--pinn-llm-dpc", default=None,
                     help="Path to LLM+DPC PINN .pt")
    ap.add_argument("--pinn-llm", default=None,
                     help="Path to LLM-only PINN .pt (no DPC)")
    ap.add_argument("--sac", default=None, help="Path to SAC .zip")
    ap.add_argument("--ddpg", default=None, help="Path to DDPG .zip")
    ap.add_argument("--ppo", default=None, help="Path to PPO .zip")
    ap.add_argument("--n-reps", type=int, default=30)
    ap.add_argument("--output", default="results/bloor_protocol_comparison.json")
    a = ap.parse_args()

    results = {}

    if a.pinn_distill and Path(a.pinn_distill).exists():
        net = _load_pinn(a.pinn_distill)
        _eval_both("PINN + NMPC Distill (ours)", _pinn_query(net),
                     a.n_reps, results)

    if a.pinn_llm_dpc and Path(a.pinn_llm_dpc).exists():
        net = _load_pinn(a.pinn_llm_dpc)
        _eval_both("LLM + Two-Phase DPC (ours, no distill)",
                     _pinn_query(net), a.n_reps, results)

    if a.pinn_llm and Path(a.pinn_llm).exists():
        net = _load_pinn(a.pinn_llm)
        _eval_both("LLM only (ours, no DPC, no distill)",
                     _pinn_query(net), a.n_reps, results)

    if a.sac and Path(a.sac).exists():
        model = _load_sb3(a.sac, "SAC")
        _eval_both("SAC (ours, 200K timesteps)", _sb3_query(model),
                     a.n_reps, results)

    if a.ddpg and Path(a.ddpg).exists():
        model = _load_sb3(a.ddpg, "DDPG")
        _eval_both("DDPG (ours, 200K timesteps)", _sb3_query(model),
                     a.n_reps, results)

    if a.ppo and Path(a.ppo).exists():
        model = _load_sb3(a.ppo, "PPO")
        _eval_both("PPO (ours, 200K timesteps)", _sb3_query(model),
                     a.n_reps, results)

    if not results:
        print("\nNo models provided / found. Pass at least one --pinn-* "
              "or --sac/--ddpg/--ppo flag.")
        return

    # Summary table
    print("\n" + "=" * 100)
    print("=== Four-tank: BOTH protocols side-by-side ===")
    print("=" * 100)
    print(f"{'Method':<42}"
          f"{'constant opt_gap':>20}{'constant MAD':>16}"
          f"{'Bloor opt_gap':>20}{'Bloor MAD':>14}")
    print("-" * 100)
    for label, protocols in results.items():
        c = protocols["constant setpoints"]
        b = protocols["Bloor step changes"]
        print(f"{label:<42}"
              f"{c['opt_gap']:>20.4f}{c['MAD']:>16.4f}"
              f"{b['opt_gap']:>20.4f}{b['MAD']:>14.4f}")

    print("\nReference: Bloor 2025 Table 5 (their 50K-timestep results)")
    print(f"{'  Bloor DDPG (their best on four-tank)':<42}"
          f"{'-':>20}{'-':>16}"
          f"{'0.0427':>20}{'0.0980':>14}")
    print(f"{'  Bloor SAC':<42}{'-':>20}{'-':>16}"
          f"{'0.0537':>20}{'0.0788':>14}")
    print(f"{'  Bloor PPO':<42}{'-':>20}{'-':>16}"
          f"{'0.0690':>20}{'0.0994':>14}")

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {a.output}")


if __name__ == "__main__":
    main()
