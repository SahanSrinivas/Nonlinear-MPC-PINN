"""Closed-loop evaluation of a trained PINN-MPC.

Ported from cell 23 / cell 26 of Kardamaki et al. 2026 SISO notebook.
The evaluation metric matches what they use (Tables 4, 5):
  - Steady-state offset Delta_y = y(t=t_end) - y_sp per episode
  - Reported as mean | min | max across N episodes
  - Lower is better

Two test suites (matching their paper, p.7):
  - Set-point tracking: random (ysp, x0, u0), d0 = 0
  - Disturbance rejection: random (ysp, x0, u0, d0) with d0 > 0
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from .pinn_siso import DEVICE, PINN_Controller


@dataclass
class EvalScenario:
    """Match Kardamaki et al. 2026 SISO scenario constants (Section 4.1.1)."""
    K_VALVE: float = 0.7        # m^0.5/s outflow coefficient
    A: float = 1.0              # m^2 tank cross-section
    dt: float = 0.01            # plant integration step (s)
    Ts: float = 1.0             # controller sampling time (s)
    sim_time: float = 30.0      # total simulation per episode (s)
    sp_change: float = 5.0      # setpoint step time (s)
    d_step: float = 30.0        # disturbance step time (s; >sim_time = no step)
    noise: float = 0.0          # measurement noise std

    # Episode sampling bounds (Section 4.1.1)
    x0_lo: float = 0.0
    x0_hi: float = 3.0
    u0_lo: float = 0.0
    u0_hi: float = 1.0
    d_lo: float = 0.0
    d_hi: float = 0.4


@torch.no_grad()
def rollout_batch(model: PINN_Controller, x0: torch.Tensor,
                    ysp: torch.Tensor, u0: torch.Tensor, d0: torch.Tensor,
                    scen: EvalScenario | None = None) -> dict:
    """Parallel closed-loop simulation (ported from cell 23 of their notebook)."""
    scen = scen or EvalScenario()
    bs = x0.shape[0]
    n_query_steps = int(scen.sim_time / scen.Ts)
    n_inner = int(scen.Ts / scen.dt)

    x_curr = x0.clone().to(DEVICE)
    x_meas = x_curr + torch.randn_like(x_curr) * scen.noise
    u_prev = u0.to(DEVICE) if u0 is not None else (
        scen.K_VALVE * torch.sqrt(x_curr.clamp(min=0.0)))
    d0 = d0.to(DEVICE)
    ysp_init = x0.clone().to(DEVICE)

    t_curr = 0.0
    t_log, X_log, U_log, YSP_log, D_log = [], [], [], [], []

    for _ in range(n_query_steps):
        ysp_k = ysp_init if t_curr < scen.sp_change else ysp.to(DEVICE)
        d_k = torch.zeros_like(d0) if t_curr < scen.d_step else d0
        t_query = torch.full((bs,), scen.Ts, device=DEVICE)

        _, u_pred = model(t_query, x_meas, u_prev, ysp_k, d_k)
        u_cmd = u_pred
        x_inner = x_curr.clone()
        for _ in range(n_inner):
            dx = ((u_cmd + d_k
                   - scen.K_VALVE * torch.sqrt(x_inner.clamp(min=0.0)))
                  / scen.A)
            x_inner = (x_inner + dx * scen.dt).clamp(min=0.0)
            t_curr += scen.dt
            t_log.append(t_curr)
            X_log.append(x_inner.cpu())
            U_log.append(u_cmd.cpu())
            YSP_log.append(ysp_k.cpu())
            D_log.append(d_k.cpu())
        x_curr = x_inner.clone()
        x_meas = (x_curr + torch.randn_like(x_curr) * scen.noise).clamp(min=0.0)
        u_prev = u_cmd.clone()

    return {
        "t": torch.tensor(t_log[::n_inner]),   # one per controller step
        "X": torch.stack(X_log[::n_inner], dim=1),     # (bs, T)
        "U": torch.stack(U_log[::n_inner], dim=1),
        "YSP": torch.stack(YSP_log[::n_inner], dim=1),
        "D": torch.stack(D_log[::n_inner], dim=1),
    }


def sample_episodes(n: int, scen: EvalScenario | None = None,
                     mode: str = "tracking", seed: int = 0) -> dict:
    """Sample n random (x0, u0, ysp, d0) per Kardamaki Section 4.1.1.
    mode: 'tracking' -> d0 = 0; 'disturbance' -> d0 sampled from [d_lo, d_hi].
    """
    scen = scen or EvalScenario()
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(scen.x0_lo, scen.x0_hi, n).astype(np.float32)
    u0 = rng.uniform(scen.u0_lo, scen.u0_hi, n).astype(np.float32)
    if mode == "disturbance":
        # 20% probability d0=0, else uniform [d_lo, d_hi] (paper p.7)
        d0 = rng.uniform(scen.d_lo, scen.d_hi, n).astype(np.float32)
        zero_mask = rng.uniform(0, 1, n) < 0.2
        d0[zero_mask] = 0.0
    else:
        d0 = np.zeros(n, dtype=np.float32)
    # Set-point range: [d/K, min(3, (1+d)/K)^2] from paper Section 4.1.1
    ysp_lo = (d0 / scen.K_VALVE) ** 2
    ysp_hi = np.minimum(3.0, ((1 + d0) / scen.K_VALVE) ** 2)
    ysp = rng.uniform(ysp_lo, ysp_hi, n).astype(np.float32)
    return {
        "x0":  torch.tensor(x0,  device=DEVICE),
        "u0":  torch.tensor(u0,  device=DEVICE),
        "ysp": torch.tensor(ysp, device=DEVICE),
        "d0":  torch.tensor(d0,  device=DEVICE),
    }


@torch.no_grad()
def evaluate_model(model: PINN_Controller, n_tracking: int = 500,
                    n_disturbance: int = 500,
                    scen: EvalScenario | None = None,
                    seed: int = 0) -> dict:
    """Run set-point tracking + disturbance rejection suites; return metrics.

    Metrics (matching Kardamaki Table 4):
      - tracking_mean_offset, tracking_max_offset
      - disturbance_mean_offset, disturbance_max_offset
      - combined_score = mean(|abs offset|) across both suites - the metric
        we'll minimize during hparam search.
    """
    scen = scen or EvalScenario()
    model.eval()

    # Set-point tracking suite (d0=0)
    eps_t = sample_episodes(n_tracking, scen, mode="tracking", seed=seed)
    res_t = rollout_batch(model, eps_t["x0"], eps_t["ysp"],
                            eps_t["u0"], eps_t["d0"], scen)
    final_x_t = res_t["X"][:, -1]
    offset_t = (final_x_t - eps_t["ysp"].cpu()).abs().numpy()
    track_mean = float(offset_t.mean())
    track_max  = float(offset_t.max())

    # Disturbance rejection suite (d0>0)
    eps_d = sample_episodes(n_disturbance, scen, mode="disturbance",
                             seed=seed + 1)
    res_d = rollout_batch(model, eps_d["x0"], eps_d["ysp"],
                            eps_d["u0"], eps_d["d0"], scen)
    final_x_d = res_d["X"][:, -1]
    offset_d = (final_x_d - eps_d["ysp"].cpu()).abs().numpy()
    dist_mean = float(offset_d.mean())
    dist_max  = float(offset_d.max())

    # Combined: mean of both means + max of both maxes (Kardamaki's eval
    # protocol minimizes both mean+max steady-state offsets)
    combined_score = 0.5 * (track_mean + dist_mean) + 0.25 * (track_max + dist_max)

    return {
        "tracking_mean_offset_m": track_mean,
        "tracking_max_offset_m":  track_max,
        "disturbance_mean_offset_m": dist_mean,
        "disturbance_max_offset_m":  dist_max,
        "combined_score": float(combined_score),
        "n_tracking": n_tracking,
        "n_disturbance": n_disturbance,
    }


if __name__ == "__main__":
    # Smoke test: train a tiny PINN, then evaluate
    from .pinn_siso import PINNHparams, train_pinn_siso

    print(f"Device: {DEVICE}")
    print("Loading SISO training samples...")
    data = torch.load(
        "C:/Pegasus-Sample/Nonlinear-LLMs-PINN-MPC/external/siso_training_samples.pt",
        map_location="cpu", weights_only=False)
    x0_all = data["x0_all"].to(DEVICE)
    u0_all = data["u0_all"].to(DEVICE)
    ysp_all = data["ysp_all"].to(DEVICE)
    d0_all = data["d0_all"].to(DEVICE)

    # Reduced-scale training
    hp = PINNHparams(K1=200, K2=200, bs=50)
    print(f"\nReduced-scale training: K={hp.K1}+{hp.K2} epochs, bs={hp.bs}...")
    t0 = time.time()
    model, _ = train_pinn_siso(hp, x0_all, u0_all, ysp_all, d0_all,
                                 verbose=False)
    train_time = time.time() - t0
    print(f"  Train time: {train_time:.1f}s")

    print("\nEvaluating (200 tracking + 200 disturbance episodes)...")
    t0 = time.time()
    metrics = evaluate_model(model, n_tracking=200, n_disturbance=200)
    eval_time = time.time() - t0
    print(f"  Eval time: {eval_time:.1f}s")
    print()
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nKardamaki paper (SISO Table 4):")
    print(f"  tracking_mean_offset_m  : 1.61e-2  (expect ours <0.1 at this scale)")
    print(f"  tracking_max_offset_m   : 3.22e-2")
    print(f"  disturbance_mean_offset_m: 1.29e-2")
    print(f"  disturbance_max_offset_m: 4.53e-2")
