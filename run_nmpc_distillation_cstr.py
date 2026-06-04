"""run_nmpc_distillation_cstr.py - NMPC distillation for the CSTR.

Mirrors run_nmpc_distillation_fourtank.py for PC-Gym Case Study 1 (CSTR).

CSTR is the SIMPLEST PC-Gym benchmark (2 states, 1 input, linear scaling),
so physics losses SHOULD be stable here (unlike crystallization, which has
log-denormed outputs spanning 10+ orders of magnitude).

Pipeline:
  1. Sample 2000 episodes WITH NMPC query_nmpc=True (records u_NMPC = T_c).
  2. Train PINN with composite_loss_cstr PLUS L_nmpc behavior cloning.
  3. Evaluate via evaluate_cstr (Bloor opt_gap + MAD).

Usage on Colab:
  !python -u run_nmpc_distillation_cstr.py \\
      --n-train-eps 2000 --K1 10000 --K2 10000 \\
      --w-nmpc 200.0 \\
      --output /content/drive/MyDrive/pinn_mpc_results/nmpc_distill_cstr_seed0 \\
      --seed 0 \\
      2>&1 | tee /content/drive/MyDrive/pinn_mpc_results/distill_cstr.log

Reference numbers (Bloor 2025 Table 5, CSTR — the EASY case):
  DDPG = 0.0005, SAC = 0.0007, PPO = 0.0006

If our PINN+Distill hits opt_gap ~0.001-0.005, we've matched Bloor's RL.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from agentic_pcgym.pinn_cstr import (
    PINN_CSTR, CSTRPINNHparams, DEVICE)
from agentic_pcgym.pinn_training import train_pinn_cstr
from agentic_pcgym.data_gen import (
    sample_cstr_episodes, save_episodes, load_episodes)
from agentic_pcgym.evaluator import evaluate_cstr


# Default config — CSTR is simple, physics losses should be stable.
CSTR_DEFAULT_CFG = {
    "w_ode":   100.0,   # physics ON (CSTR is well-conditioned)
    "w_ic":    10.0,
    "w_ytrk":  10.0,
    "w_utrk":  1.0,
    "w_du":    1.0,
    "w_u":     100.0,
    "w_x":     10.0,
    "lr1":     1e-3,
    "lr2":     2e-4,
}


@contextlib.contextmanager
def suppress_fortran_stderr():
    """Silence Fortran/C++ stderr (LSODA, CasADi) during noisy operations."""
    old_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stderr_fd)


def _pinn_query_factory(net, t_c_min=295.0, t_c_max=302.0):
    @torch.no_grad()
    def pinn_query(C_A, T, CA_sp, T_c_prev):
        z = lambda v: torch.tensor([float(v)], device=DEVICE)
        # Use T_c_prev as T_c_ic in network input
        _, _, T_c = net(z(1.0), z(C_A), z(T), z(CA_sp), z(T_c_prev))
        return float(max(t_c_min, min(t_c_max, T_c.item())))
    return pinn_query


def _eval(net, label, out_acc, n_reps=30):
    print(f"\n[Evaluating {label} on {n_reps} closed-loop reps]")
    q = _pinn_query_factory(net)
    with suppress_fortran_stderr():
        metrics = evaluate_cstr(q, n_reps=n_reps, seed=42, verbose=False)
    print(f"  median_reward_pi:     {metrics['median_reward_pi']:>10.4f}")
    print(f"  median_reward_oracle: {metrics['median_reward_oracle']:>10.4f}")
    print(f"  optimality_gap:       {metrics['optimality_gap']:>10.4f}")
    print(f"  MAD:                  {metrics['MAD']:>10.4f}")
    out_acc[label] = metrics
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-eps", type=int, default=2000)
    ap.add_argument("--K1", type=int, default=10000)
    ap.add_argument("--K2", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--w-nmpc", type=float, default=200.0)
    ap.add_argument("--output", default="results/nmpc_distill_cstr/")
    ap.add_argument("--episodes-cache", default=None)
    ap.add_argument("--n-eval-reps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Sample episodes with NMPC labels ---
    cache = a.episodes_cache or str(out_dir / "episodes_with_nmpc.pt")
    if Path(cache).exists():
        print(f"=== Loading cached episodes from {cache} ===")
        episodes = load_episodes(cache)
        if episodes.get("u_nmpc") is None:
            raise RuntimeError(f"Cached episodes at {cache} have no u_nmpc")
        print(f"  {len(episodes['C_A_all'])} episodes loaded\n")
    else:
        print(f"=== Sampling {a.n_train_eps} CSTR episodes WITH NMPC ===")
        t0 = time.time()
        with suppress_fortran_stderr():
            episodes = sample_cstr_episodes(
                N=a.n_train_eps, seed=a.seed, query_nmpc=True, verbose=True)
        print(f"  done in {time.time()-t0:.1f}s")
        save_episodes(episodes, cache)
        print(f"  cached to {cache}\n")

    metrics_log = {}

    # --- 2. Train PINN ---
    print(f"\n=== Training PINN with NMPC distillation (w_nmpc={a.w_nmpc}) ===")
    hp = CSTRPINNHparams(
        K1=a.K1, K2=a.K2, bs=a.bs,
        w_nmpc=a.w_nmpc,
        **{k: float(CSTR_DEFAULT_CFG[k]) for k in
           ["w_ode", "w_ic", "w_ytrk", "w_utrk", "w_du", "w_u", "w_x",
            "lr1", "lr2"]})
    print(f"  cfg: w_ode={hp.w_ode}, w_ytrk={hp.w_ytrk}, "
          f"w_nmpc={hp.w_nmpc}, lr1={hp.lr1:.1e}")
    net = PINN_CSTR().to(DEVICE)
    t0 = time.time()
    hist = train_pinn_cstr(net, episodes, hp, verbose=True, seed=a.seed)
    print(f"\n  trained in {time.time()-t0:.1f}s")
    if hist.get("nan_at"):
        print(f"  WARNING: NaN at {hist['nan_at']}")
    torch.save(net.state_dict(), out_dir / "pinn_cstr_distill.pt")

    _eval(net, "DISTILLED CSTR (PINN + NMPC BC + physics)", metrics_log,
          n_reps=a.n_eval_reps)

    # --- 3. Summary ---
    print("\n" + "=" * 78)
    print("=== CSTR NMPC Distillation Results ===")
    print("=" * 78)
    print(f"{'stage':<55}{'opt gap':>12}{'MAD':>12}")
    print("-" * 79)
    for label, m in metrics_log.items():
        print(f"  {label:<53}{m['optimality_gap']:>12.4f}"
              f"{m['MAD']:>12.4f}")
    print("\nReference (Bloor 2025 Table 5, CSTR):")
    print(f"  DDPG: 0.0005    SAC: 0.0007    PPO: 0.0006")

    summary = {
        "case": "cstr_nmpc_distillation",
        "cfg": {k: float(getattr(hp, k)) for k in [
            "w_ode", "w_ic", "w_ytrk", "w_utrk", "w_du", "w_u", "w_x",
            "lr1", "lr2", "w_nmpc", "K1", "K2"]},
        "n_train_eps": a.n_train_eps, "seed": a.seed,
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
