"""DPC (Differentiable Closed-loop) refinement for the PC-Gym PINNs.

Same idea as `agentic_pinn_mpc/rl_refine.py` (the single-tank version) but
adapted to multi-state plants with MIMO inputs.

Method: After Kardamaki-style PINN training (per-episode supervised loss),
refine the network weights by ROLLING OUT the PINN as a closed-loop
controller, computing trajectory-level closed-loop cost, and backpropagating
through the differentiable plant + PINN.

For each plant we differentiate through:
  - The PINN's forward pass (already differentiable)
  - Euler integration of the plant ODE (smooth and differentiable)
  - Composite trajectory cost (tracking + smoothness + bounds)

References (carried over from single-tank version):
  [1] Drgona et al. 2022, J. Process Control 116, 80-92.  (DPC)
  [2] Kardamaki et al. 2026, J. Process Control 158, 103634.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .pinn_crystallization import (
    PINN_Crystallization, T_C_LO, T_C_HI, DEVICE)
from .pinn_fourtank import (
    PINN_FourTank, V_LO, V_HI, H_LO, H_HI)
from .plants.crystallization import CrystParams, CrystScenario
from .plants.fourtank import FourTankParams, FourTankScenario


# ============================================================================
# Crystallization: differentiable closed-loop rollout
# ============================================================================
def differentiable_rollout_crystallization(
        net: PINN_Crystallization,
        x0: torch.Tensor,    # (B, 5) - [mu_0, mu_1, mu_2, mu_3, c]
        cv_sp: torch.Tensor, # (B,)
        ln_sp: torch.Tensor, # (B,)
        Tc_init: torch.Tensor,  # (B,)
        n_steps: int = 30,
        dt: float = 1.0,
        n_inner: int = 5,     # internal Euler sub-steps per controller dt
        params: CrystParams | None = None,
        ) -> dict:
    """Closed-loop rollout keeping the autograd graph.

    Returns dict with trajectories (X (B,T,5), U (B,T)) and final-state
    tracking offset (CV, L_n).
    """
    p = params or CrystParams()
    B = x0.shape[0]
    sub_dt = dt / n_inner

    mu0, mu1, mu2, mu3, c = x0[:, 0], x0[:, 1], x0[:, 2], x0[:, 3], x0[:, 4]
    Tc_prev = Tc_init.clone()
    X_log, U_log = [], []

    for k in range(n_steps):
        # Query PINN at t=dt (the predicted "next control")
        t_query = torch.full((B,), dt, device=DEVICE)
        _, _, _, _, _, Tc_cmd = net(t_query, mu0, mu1, mu2, mu3, c,
                                       cv_sp, ln_sp, Tc_prev)
        # Clip via differentiable saturation (use clamp; gradient is 0 outside)
        Tc_cmd = Tc_cmd.clamp(T_C_LO, T_C_HI)
        # Inner Euler integration
        for _ in range(n_inner):
            T_K = Tc_cmd + 273.15
            Ceq = -686.2686 + 3.579165 * T_K - 0.00292874 * T_K * T_K
            S = c * 1000.0 - Ceq
            S2 = (S * S).clamp(min=0.0)
            mu3_sq = (mu3 * mu3).clamp(min=0.0)
            B_0 = (p.k_a * torch.exp(p.k_b / T_K)
                   * (S2 ** (p.k_c / 2.0))
                   * (mu3_sq ** (p.k_d / 2.0)))
            G_inf = (p.k_g * torch.exp(p.k_1 / T_K)
                     * (S2 ** (p.k_2 / 2.0)))
            dmu0 = B_0
            dmu1 = G_inf * (p.a * mu0 + p.b * mu1 * 1e-4) * 1e4
            dmu2 = 2.0 * G_inf * (p.a * mu1*1e-4 + p.b * mu2*1e-8) * 1e8
            dmu3 = 3.0 * G_inf * (p.a * mu2*1e-8 + p.b * mu3*1e-12) * 1e12
            dc   = -0.5 * p.rho * p.alpha * G_inf * (p.a * mu2*1e-8
                                                       + p.b * mu3*1e-12)
            mu0 = (mu0 + sub_dt * dmu0).clamp(min=0.0)
            mu1 = (mu1 + sub_dt * dmu1).clamp(min=0.0)
            mu2 = (mu2 + sub_dt * dmu2).clamp(min=0.0)
            mu3 = (mu3 + sub_dt * dmu3).clamp(min=0.0)
            c   = (c + sub_dt * dc).clamp(min=0.0)
        X_log.append(torch.stack([mu0, mu1, mu2, mu3, c], dim=1))
        U_log.append(Tc_cmd)
        Tc_prev = Tc_cmd

    X = torch.stack(X_log, dim=1)     # (B, T, 5)
    U = torch.stack(U_log, dim=1)     # (B, T)
    # Final CV, L_n
    mu0_f, mu1_f, mu2_f = X[:, -1, 0], X[:, -1, 1], X[:, -1, 2]
    CV_f = torch.sqrt(torch.clamp(mu2_f * mu0_f / (mu1_f*mu1_f + 1e-30) - 1.0,
                                    min=0.0))
    Ln_f = mu1_f / (mu0_f + 1e-30)
    cv_err = CV_f - cv_sp
    ln_err = (Ln_f - ln_sp) / 15.0   # normalised
    final_offset = cv_err.pow(2) + ln_err.pow(2)
    return {"X": X, "U": U, "CV_final": CV_f, "Ln_final": Ln_f,
             "offset_final": final_offset}


# ============================================================================
# Four-tank: differentiable closed-loop rollout (MIMO)
# ============================================================================
def differentiable_rollout_fourtank(
        net: PINN_FourTank,
        x0: torch.Tensor,                   # (B, 4)
        h1_sp: torch.Tensor, h2_sp: torch.Tensor,   # (B,)
        v1_init: torch.Tensor, v2_init: torch.Tensor,   # (B,)
        n_steps: int = 60,
        dt: float = 1000.0 / 60.0,
        n_inner: int = 5,
        params: FourTankParams | None = None,
        ) -> dict:
    p = params or FourTankParams()
    B = x0.shape[0]
    sub_dt = dt / n_inner
    h1, h2, h3, h4 = x0[:, 0], x0[:, 1], x0[:, 2], x0[:, 3]
    v1_prev, v2_prev = v1_init.clone(), v2_init.clone()
    H_log, V_log = [], []
    twog = 2.0 * p.g_a
    eps = 1e-6
    for k in range(n_steps):
        t_query = torch.full((B,), dt, device=DEVICE)
        _, _, _, _, v1_cmd, v2_cmd = net(t_query, h1, h2, h3, h4,
                                            h1_sp, h2_sp, v1_prev, v2_prev)
        v1_cmd = v1_cmd.clamp(V_LO, V_HI)
        v2_cmd = v2_cmd.clamp(V_LO, V_HI)
        for _ in range(n_inner):
            s1 = torch.sqrt(h1.clamp(min=0.0) * twog + eps)
            s2 = torch.sqrt(h2.clamp(min=0.0) * twog + eps)
            s3 = torch.sqrt(h3.clamp(min=0.0) * twog + eps)
            s4 = torch.sqrt(h4.clamp(min=0.0) * twog + eps)
            dh1 = (-(p.a_1/p.A_1)*s1 + (p.a_3/p.A_1)*s3
                   + (p.gamma_1*p.k_1/p.A_1)*v1_cmd)
            dh2 = (-(p.a_2/p.A_2)*s2 + (p.a_4/p.A_2)*s4
                   + (p.gamma_2*p.k_2/p.A_2)*v2_cmd)
            dh3 = (-(p.a_3/p.A_3)*s3
                   + ((1-p.gamma_2)*p.k_2/p.A_3)*v2_cmd)
            dh4 = (-(p.a_4/p.A_4)*s4
                   + ((1-p.gamma_1)*p.k_1/p.A_4)*v1_cmd)
            h1 = (h1 + sub_dt * dh1).clamp(min=0.0)
            h2 = (h2 + sub_dt * dh2).clamp(min=0.0)
            h3 = (h3 + sub_dt * dh3).clamp(min=0.0)
            h4 = (h4 + sub_dt * dh4).clamp(min=0.0)
        H_log.append(torch.stack([h1, h2, h3, h4], dim=1))
        V_log.append(torch.stack([v1_cmd, v2_cmd], dim=1))
        v1_prev, v2_prev = v1_cmd, v2_cmd
    H = torch.stack(H_log, dim=1)   # (B, T, 4)
    V = torch.stack(V_log, dim=1)   # (B, T, 2)
    h1_err = H[:, -1, 0] - h1_sp
    h2_err = H[:, -1, 1] - h2_sp
    return {"H": H, "V": V, "h1_final": H[:, -1, 0], "h2_final": H[:, -1, 1],
             "offset_final": h1_err.pow(2) + h2_err.pow(2)}


# ============================================================================
# DPC refinement training loops
# ============================================================================
@dataclass
class DPCRefineCfg:
    epochs: int = 200
    bs: int = 16
    lr: float = 1e-5             # tiny - refinement, don't disrupt training
    w_offset: float = 100.0      # final-state tracking
    w_traj: float = 1.0          # full-trajectory tracking
    w_smooth: float = 0.01       # input smoothness
    w_bounds: float = 10.0       # state-bound soft penalty
    early_stop_patience: int = 20
    seed: int = 0


def refine_crystallization_dpc(
        net: PINN_Crystallization,
        cfg: DPCRefineCfg | None = None,
        params: CrystParams | None = None,
        verbose: bool = False,
        ) -> dict:
    """Differentiable closed-loop refinement for crystallization PINN."""
    from .data_gen import sample_crystallization_episodes
    cfg = cfg or DPCRefineCfg()
    params = params or CrystParams()
    torch.manual_seed(cfg.seed)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    hist = {"loss": [], "offset": []}
    best, bad = float("inf"), 0
    for ep in range(1, cfg.epochs + 1):
        eps = sample_crystallization_episodes(
            N=cfg.bs, seed=ep, query_nmpc=False, verbose=False)
        x0 = torch.stack([eps[k] for k in ["mu0_all", "mu1_all", "mu2_all",
                                              "mu3_all", "c_all"]], dim=1).to(DEVICE)
        cv_sp = eps["cv_sp_all"].to(DEVICE)
        ln_sp = eps["ln_sp_all"].to(DEVICE)
        Tc_init = eps["Tc_all"].to(DEVICE)
        roll = differentiable_rollout_crystallization(
            net, x0, cv_sp, ln_sp, Tc_init,
            n_steps=CrystScenario().n_steps, dt=CrystScenario().dt_hr,
            params=params)
        offset = roll["offset_final"].mean()
        # Trajectory smoothness
        dU = (roll["U"][:, 1:] - roll["U"][:, :-1]) / (T_C_HI - T_C_LO)
        smooth = dU.pow(2).mean()
        # State bounds (non-negative)
        bound = torch.relu(-roll["X"]).pow(2).mean()
        loss = (cfg.w_offset * offset
                 + cfg.w_smooth * smooth
                 + cfg.w_bounds * bound)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        hist["loss"].append(float(loss.item()))
        hist["offset"].append(float(offset.item()))
        if verbose and (ep == 1 or ep % 20 == 0):
            print(f"  DPC ep {ep:4d}: loss={loss.item():.4e}, offset={offset.item():.4f}")
        if loss.item() < best - 1e-6:
            best = loss.item(); bad = 0
        else:
            bad += 1
            if bad >= cfg.early_stop_patience:
                if verbose:
                    print(f"  Early stop at ep {ep}")
                break
    net.eval()
    return hist


def refine_fourtank_dpc(
        net: PINN_FourTank,
        cfg: DPCRefineCfg | None = None,
        params: FourTankParams | None = None,
        verbose: bool = False,
        ) -> dict:
    from .data_gen import sample_fourtank_episodes
    cfg = cfg or DPCRefineCfg()
    params = params or FourTankParams()
    torch.manual_seed(cfg.seed)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    hist = {"loss": [], "offset": []}
    best, bad = float("inf"), 0
    for ep in range(1, cfg.epochs + 1):
        eps = sample_fourtank_episodes(N=cfg.bs, seed=ep, query_nmpc=False,
                                          verbose=False)
        x0 = torch.stack([eps[k] for k in ["h1_all", "h2_all", "h3_all",
                                              "h4_all"]], dim=1).to(DEVICE)
        h1_sp = eps["h1_sp_all"].to(DEVICE)
        h2_sp = eps["h2_sp_all"].to(DEVICE)
        v1_init = eps["v1_all"].to(DEVICE)
        v2_init = eps["v2_all"].to(DEVICE)
        roll = differentiable_rollout_fourtank(
            net, x0, h1_sp, h2_sp, v1_init, v2_init,
            n_steps=FourTankScenario().n_steps,
            dt=FourTankScenario().dt_s, params=params)
        offset = roll["offset_final"].mean()
        # Smoothness
        dV = (roll["V"][:, 1:] - roll["V"][:, :-1]) / (V_HI - V_LO)
        smooth = dV.pow(2).mean()
        bound = torch.relu(-roll["H"]).pow(2).mean()
        loss = (cfg.w_offset * offset
                 + cfg.w_smooth * smooth
                 + cfg.w_bounds * bound)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        hist["loss"].append(float(loss.item()))
        hist["offset"].append(float(offset.item()))
        if verbose and (ep == 1 or ep % 20 == 0):
            print(f"  DPC ep {ep:4d}: loss={loss.item():.4e}, offset={offset.item():.4f}")
        if loss.item() < best - 1e-6:
            best = loss.item(); bad = 0
        else:
            bad += 1
            if bad >= cfg.early_stop_patience:
                if verbose:
                    print(f"  Early stop at ep {ep}")
                break
    net.eval()
    return hist
