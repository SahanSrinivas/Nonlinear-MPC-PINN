"""Shared PINN training + loss for the PC-Gym case studies.

Implements the composite Kardamaki-style loss adapted to:
  - Crystallization (5 states, 1 input, 2 outputs CV+L_n)
  - Four-tank (4 states, 2 inputs, 2 outputs h_1+h_2)

The training loop is two-phase Adam (Kardamaki Sec 3.3 schedule):
  Phase 1: w_ode + w_ic + w_ytrk + w_utrk + (w_xtrk for four-tank)
  Phase 2: add w_du + w_u + w_x (constraint terms)

References:
  - Kardamaki et al. 2026, J. Process Control 158, 103634
  - Bloor et al. 2025, Comp & Chem Eng 204, 109363  (PC-Gym)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from .pinn_crystallization import (
    PINN_Crystallization, CrystPINNHparams, DEVICE,
    T_C_LO, T_C_HI)
from .pinn_fourtank import (
    PINN_FourTank, FourTankPINNHparams, V_LO, V_HI, H_LO, H_HI)
from .pinn_cstr import (
    PINN_CSTR, CSTRPINNHparams,
    T_C_LO as CSTR_TC_LO, T_C_HI as CSTR_TC_HI,
    C_A_LO, C_A_HI, T_LO as CSTR_T_LO, T_HI as CSTR_T_HI)
from .plants.crystallization import CrystParams, C_eq, CV_from_moments, Ln_from_moments
from .plants.fourtank import FourTankParams
from .plants.cstr import CSTRParams


# ============================================================================
# Crystallization-specific physics residual (ODE error)
# ============================================================================
def crystallization_ode_residual(
        net: PINN_Crystallization,
        t_col: torch.Tensor,           # (N_col,) - requires grad
        mu0_ic: torch.Tensor,          # (B,)
        mu1_ic: torch.Tensor,
        mu2_ic: torch.Tensor,
        mu3_ic: torch.Tensor,
        c_ic:   torch.Tensor,
        cv_sp:  torch.Tensor,
        ln_sp:  torch.Tensor,
        Tc_ic:  torch.Tensor,
        p: CrystParams,
        ) -> torch.Tensor:
    """Returns ODE-residual squared, averaged across (B, N_col)."""
    B = mu0_ic.shape[0]
    N_col = t_col.shape[0]
    # Repeat conditioning over collocation points
    t_flat = t_col.repeat(B).requires_grad_(True)
    mu0_f  = mu0_ic.repeat_interleave(N_col)
    mu1_f  = mu1_ic.repeat_interleave(N_col)
    mu2_f  = mu2_ic.repeat_interleave(N_col)
    mu3_f  = mu3_ic.repeat_interleave(N_col)
    c_f    = c_ic.repeat_interleave(N_col)
    cv_f   = cv_sp.repeat_interleave(N_col)
    ln_f   = ln_sp.repeat_interleave(N_col)
    Tc_f   = Tc_ic.repeat_interleave(N_col)

    # Forward pass
    mu0_p, mu1_p, mu2_p, mu3_p, c_p, Tc_p = net(
        t_flat, mu0_f, mu1_f, mu2_f, mu3_f, c_f, cv_f, ln_f, Tc_f)

    # Time derivatives via autograd
    ones = torch.ones_like(mu0_p)
    dmu0 = grad(mu0_p, t_flat, ones, create_graph=True, retain_graph=True)[0]
    dmu1 = grad(mu1_p, t_flat, ones, create_graph=True, retain_graph=True)[0]
    dmu2 = grad(mu2_p, t_flat, ones, create_graph=True, retain_graph=True)[0]
    dmu3 = grad(mu3_p, t_flat, ones, create_graph=True, retain_graph=True)[0]
    dc   = grad(c_p,   t_flat, ones, create_graph=True, retain_graph=True)[0]

    # Plant RHS (CasADi-free torch version)
    T_K = Tc_p + 273.15
    Ceq = -686.2686 + 3.579165 * T_K - 0.00292874 * T_K * T_K
    S = c_p * 1000.0 - Ceq
    S2 = S * S
    mu3_sq = mu3_p * mu3_p
    B_0 = (p.k_a * torch.exp(p.k_b / T_K)
           * (S2.clamp(min=0.0) ** (p.k_c / 2.0))
           * (mu3_sq.clamp(min=0.0) ** (p.k_d / 2.0)))
    G_inf = (p.k_g * torch.exp(p.k_1 / T_K)
             * (S2.clamp(min=0.0) ** (p.k_2 / 2.0)))
    rhs_mu0 = B_0
    rhs_mu1 = G_inf * (p.a * mu0_p + p.b * mu1_p * 1e-4) * 1e4
    rhs_mu2 = 2.0 * G_inf * (p.a * mu1_p * 1e-4 + p.b * mu2_p * 1e-8) * 1e8
    rhs_mu3 = 3.0 * G_inf * (p.a * mu2_p * 1e-8 + p.b * mu3_p * 1e-12) * 1e12
    rhs_c   = -0.5 * p.rho * p.alpha * G_inf * (p.a * mu2_p * 1e-8
                                                  + p.b * mu3_p * 1e-12)

    # Residual (normalised by typical magnitudes via log-scaling)
    # We scale residuals by their reference magnitudes to balance terms
    r_mu0 = (dmu0 - rhs_mu0) / (mu0_p.abs() + 1.0)
    r_mu1 = (dmu1 - rhs_mu1) / (mu1_p.abs() + 1.0)
    r_mu2 = (dmu2 - rhs_mu2) / (mu2_p.abs() + 1.0)
    r_mu3 = (dmu3 - rhs_mu3) / (mu3_p.abs() + 1.0)
    r_c   = (dc   - rhs_c)
    res = r_mu0**2 + r_mu1**2 + r_mu2**2 + r_mu3**2 + r_c**2
    return res.mean()


# ============================================================================
# Four-tank ODE residual
# ============================================================================
def fourtank_ode_residual(
        net: PINN_FourTank,
        t_col: torch.Tensor,
        h1_ic: torch.Tensor,
        h2_ic: torch.Tensor,
        h3_ic: torch.Tensor,
        h4_ic: torch.Tensor,
        h1_sp: torch.Tensor,
        h2_sp: torch.Tensor,
        v1_ic: torch.Tensor,
        v2_ic: torch.Tensor,
        p: FourTankParams,
        ) -> torch.Tensor:
    B = h1_ic.shape[0]
    N_col = t_col.shape[0]
    t_flat = t_col.repeat(B).requires_grad_(True)
    h1_f = h1_ic.repeat_interleave(N_col)
    h2_f = h2_ic.repeat_interleave(N_col)
    h3_f = h3_ic.repeat_interleave(N_col)
    h4_f = h4_ic.repeat_interleave(N_col)
    sp1f = h1_sp.repeat_interleave(N_col)
    sp2f = h2_sp.repeat_interleave(N_col)
    v1_f = v1_ic.repeat_interleave(N_col)
    v2_f = v2_ic.repeat_interleave(N_col)

    h1_p, h2_p, h3_p, h4_p, v1_p, v2_p = net(
        t_flat, h1_f, h2_f, h3_f, h4_f, sp1f, sp2f, v1_f, v2_f)

    ones = torch.ones_like(h1_p)
    dh1 = grad(h1_p, t_flat, ones, create_graph=True, retain_graph=True)[0]
    dh2 = grad(h2_p, t_flat, ones, create_graph=True, retain_graph=True)[0]
    dh3 = grad(h3_p, t_flat, ones, create_graph=True, retain_graph=True)[0]
    dh4 = grad(h4_p, t_flat, ones, create_graph=True, retain_graph=True)[0]

    # Plant RHS (Eqs 31-34)
    eps = 1e-6
    twog = 2.0 * p.g_a
    s1 = torch.sqrt(h1_p.clamp(min=0.0) * twog + eps)
    s2 = torch.sqrt(h2_p.clamp(min=0.0) * twog + eps)
    s3 = torch.sqrt(h3_p.clamp(min=0.0) * twog + eps)
    s4 = torch.sqrt(h4_p.clamp(min=0.0) * twog + eps)
    rhs_h1 = (-(p.a_1/p.A_1)*s1 + (p.a_3/p.A_1)*s3
              + (p.gamma_1*p.k_1/p.A_1)*v1_p)
    rhs_h2 = (-(p.a_2/p.A_2)*s2 + (p.a_4/p.A_2)*s4
              + (p.gamma_2*p.k_2/p.A_2)*v2_p)
    rhs_h3 = (-(p.a_3/p.A_3)*s3
              + ((1.0-p.gamma_2)*p.k_2/p.A_3)*v2_p)
    rhs_h4 = (-(p.a_4/p.A_4)*s4
              + ((1.0-p.gamma_1)*p.k_1/p.A_4)*v1_p)
    res = (dh1 - rhs_h1)**2 + (dh2 - rhs_h2)**2 + (dh3 - rhs_h3)**2 + (dh4 - rhs_h4)**2
    return res.mean()


# ============================================================================
# Time-grid generators
# ============================================================================
def make_time_grids(T_horizon: float, Ts: float, dt_dense: float = None,
                      dt_medium: float = None, dt_sparse: float = None):
    """Returns (t_wp, t_col) - tracking points + collocation points.

    Defaults are Kardamaki's non-uniform spacing if not provided.
    """
    # Tracking points uniformly spaced at Ts
    t_wp = torch.cat([torch.arange(0, T_horizon, Ts, device=DEVICE),
                       torch.tensor([T_horizon], device=DEVICE)])
    if dt_dense is None:
        dt_dense = T_horizon * 0.0005
        dt_medium = T_horizon * 0.005
        dt_sparse = T_horizon * 0.05
    # Non-uniform collocation (dense near t=0)
    t_back_sparse = torch.arange(T_horizon, 2*T_horizon/3, -dt_sparse, device=DEVICE)
    t_back_medium = torch.arange(2*T_horizon/3, T_horizon/3, -dt_medium, device=DEVICE)
    t_back_dense  = torch.arange(T_horizon/3, 0.0, -dt_dense, device=DEVICE)
    t_col = torch.cat([t_back_sparse, t_back_medium, t_back_dense,
                       torch.tensor([T_horizon], device=DEVICE)])
    t_col = torch.unique(t_col)
    t_col = torch.sort(t_col).values
    if t_col[0] > 0:
        t_col = torch.cat([torch.tensor([0.0], device=DEVICE), t_col])
    return t_wp, t_col


# ============================================================================
# Composite loss for Crystallization (8 weighted terms, Kardamaki-style)
# ============================================================================
def composite_loss_crystallization(
        net: PINN_Crystallization,
        t_wp: torch.Tensor, t_col: torch.Tensor,
        # batched conditioning inputs (B,)
        mu0_ic, mu1_ic, mu2_ic, mu3_ic, c_ic, cv_sp, ln_sp, Tc_ic,
        # weights
        w_ode, w_ic, w_ytrk, w_utrk, w_du, w_u, w_x,
        p: CrystParams,
        u_nmpc_batch: torch.Tensor | None = None,
        w_nmpc: float = 0.0) -> tuple[torch.Tensor, dict]:
    """Full composite loss for the crystallization PINN.

    Returns (total_loss, components_dict).

    NMPC distillation: if u_nmpc_batch is provided and w_nmpc > 0, adds an
    L_nmpc term matching PINN's T_c prediction at t=1.0 to the NMPC oracle.
    """
    B = mu0_ic.shape[0]
    N_WP = t_wp.shape[0]
    _zero = torch.tensor(0.0, device=mu0_ic.device)

    # ---- 1. ODE residual (skip if w_ode=0 to avoid NaN from log_denorm chain) ----
    if w_ode > 0.0:
        L_ode = crystallization_ode_residual(
            net, t_col, mu0_ic, mu1_ic, mu2_ic, mu3_ic, c_ic,
            cv_sp, ln_sp, Tc_ic, p)
    else:
        L_ode = _zero

    # ---- 2-8. tracking/IC/bounds (skip multi-waypoint forward if all zero) ----
    need_waypoints = any(w > 0.0 for w in
                          (w_ic, w_ytrk, w_utrk, w_du, w_u, w_x))
    if need_waypoints:
        t_flat = t_wp.repeat(B)
        mu0_f = mu0_ic.repeat_interleave(N_WP)
        mu1_f = mu1_ic.repeat_interleave(N_WP)
        mu2_f = mu2_ic.repeat_interleave(N_WP)
        mu3_f = mu3_ic.repeat_interleave(N_WP)
        c_f   = c_ic.repeat_interleave(N_WP)
        cv_f  = cv_sp.repeat_interleave(N_WP)
        ln_f  = ln_sp.repeat_interleave(N_WP)
        Tc_f  = Tc_ic.repeat_interleave(N_WP)
        mu0_p, mu1_p, mu2_p, mu3_p, c_p, Tc_p = net(
            t_flat, mu0_f, mu1_f, mu2_f, mu3_f, c_f, cv_f, ln_f, Tc_f)
        mu0_w = mu0_p.view(B, N_WP); mu1_w = mu1_p.view(B, N_WP)
        mu2_w = mu2_p.view(B, N_WP); mu3_w = mu3_p.view(B, N_WP)
        c_w   = c_p.view(B, N_WP);    Tc_w = Tc_p.view(B, N_WP)
        # CV and L_n at each tracking point (paper Eqs 29, repo Ln=mu1/mu0)
        CV_w = torch.sqrt(torch.clamp(mu2_w * mu0_w / (mu1_w*mu1_w + 1e-30) - 1.0,
                                         min=0.0))
        Ln_w = mu1_w / (mu0_w + 1e-30)

        # ---- 2. IC: state at t=0 matches IC ----
        L_ic = ((mu0_w[:, 0] - mu0_ic) / (mu0_ic.abs() + 1.0)).pow(2).mean()
        L_ic = L_ic + ((mu1_w[:, 0] - mu1_ic) / (mu1_ic.abs() + 1.0)).pow(2).mean()
        L_ic = L_ic + ((mu2_w[:, 0] - mu2_ic) / (mu2_ic.abs() + 1.0)).pow(2).mean()
        L_ic = L_ic + ((mu3_w[:, 0] - mu3_ic) / (mu3_ic.abs() + 1.0)).pow(2).mean()
        L_ic = L_ic + (c_w[:, 0] - c_ic).pow(2).mean()

        # ---- 3. Output tracking: CV -> cv_sp, L_n -> ln_sp ----
        cv_sp_b = cv_sp.unsqueeze(1)
        ln_sp_b = ln_sp.unsqueeze(1)
        L_ytrk = (((CV_w - cv_sp_b)).pow(2).mean()
                  + ((Ln_w - ln_sp_b) / 15.0).pow(2).mean())

        # ---- 4. Input tracking ----
        L_utrk = ((Tc_w - Tc_ic.unsqueeze(1)) / (T_C_HI - T_C_LO)).pow(2).mean()

        # ---- 5. Move suppression ----
        dT = (Tc_w[:, 1:] - Tc_w[:, :-1]) / (T_C_HI - T_C_LO)
        L_du = (dT.abs() - 0.1).clamp(min=0.0).pow(2).mean()

        # ---- 6. Input bounds ----
        L_u = (F.relu(T_C_LO - Tc_w).pow(2) + F.relu(Tc_w - T_C_HI).pow(2)).mean()

        # ---- 7. State bounds: c >= 0 ----
        L_x = F.relu(-c_w).pow(2).mean()
    else:
        # Pure-distillation mode: all waypoint-dependent losses skipped.
        L_ic = L_ytrk = L_utrk = L_du = L_u = L_x = _zero

    # ---- 8. NMPC behavior-cloning loss (only if u_nmpc provided) ----
    # Match PINN's T_c at t=1.0 (same query the controller uses) to NMPC oracle
    L_nmpc = torch.tensor(0.0, device=mu0_ic.device)
    if u_nmpc_batch is not None and w_nmpc > 0.0:
        t_query = torch.full_like(mu0_ic, 1.0)
        _, _, _, _, _, Tc_q = net(
            t_query, mu0_ic, mu1_ic, mu2_ic, mu3_ic, c_ic,
            cv_sp, ln_sp, Tc_ic)
        L_nmpc = ((Tc_q - u_nmpc_batch) / (T_C_HI - T_C_LO)).pow(2).mean()

    total = (w_ode * L_ode + w_ic * L_ic + w_ytrk * L_ytrk + w_utrk * L_utrk
              + w_du * L_du + w_u * L_u + w_x * L_x
              + w_nmpc * L_nmpc)
    components = {"L_ode": L_ode, "L_ic": L_ic, "L_ytrk": L_ytrk,
                   "L_utrk": L_utrk, "L_du": L_du, "L_u": L_u, "L_x": L_x,
                   "L_nmpc": L_nmpc}
    return total, components


# ============================================================================
# Composite loss for Four-tank (MIMO, with w_xtrk for upper tanks)
# ============================================================================
def composite_loss_fourtank(
        net: PINN_FourTank,
        t_wp: torch.Tensor, t_col: torch.Tensor,
        h1_ic, h2_ic, h3_ic, h4_ic, h1_sp, h2_sp, v1_ic, v2_ic,
        w_ode, w_ic, w_ytrk, w_xtrk, w_utrk, w_du, w_u,
        p: FourTankParams,
        u_nmpc_batch: torch.Tensor | None = None,
        w_nmpc: float = 0.0) -> tuple[torch.Tensor, dict]:
    B = h1_ic.shape[0]
    N_WP = t_wp.shape[0]

    L_ode = fourtank_ode_residual(net, t_col,
        h1_ic, h2_ic, h3_ic, h4_ic, h1_sp, h2_sp, v1_ic, v2_ic, p)

    t_flat = t_wp.repeat(B)
    def _rep(z): return z.repeat_interleave(N_WP)
    h1_p, h2_p, h3_p, h4_p, v1_p, v2_p = net(
        t_flat, _rep(h1_ic), _rep(h2_ic), _rep(h3_ic), _rep(h4_ic),
        _rep(h1_sp), _rep(h2_sp), _rep(v1_ic), _rep(v2_ic))
    h1_w, h2_w, h3_w, h4_w, v1_w, v2_w = [
        x.view(B, N_WP) for x in (h1_p, h2_p, h3_p, h4_p, v1_p, v2_p)]

    # 2. IC
    L_ic = ((h1_w[:, 0] - h1_ic).pow(2).mean()
             + (h2_w[:, 0] - h2_ic).pow(2).mean()
             + (h3_w[:, 0] - h3_ic).pow(2).mean()
             + (h4_w[:, 0] - h4_ic).pow(2).mean())

    # 3. Output tracking h_1, h_2
    sp1 = h1_sp.unsqueeze(1); sp2 = h2_sp.unsqueeze(1)
    L_ytrk = ((h1_w - sp1).pow(2).mean() + (h2_w - sp2).pow(2).mean())

    # 4. Upper-tank steady-state tracking (Kardamaki Sec 4.2 adds this)
    # Steady-state h_3 = ((1-gamma_2)*k_2*v2_sp / a_3)^2 / (2*g)  (analytical)
    # For training, approximate v2_sp ~ v2_ic; loose anchor.
    twog = 2.0 * p.g_a
    h3_target = ((1.0 - p.gamma_2) * p.k_2 * v2_ic.unsqueeze(1) / p.a_3) ** 2 / twog
    h4_target = ((1.0 - p.gamma_1) * p.k_1 * v1_ic.unsqueeze(1) / p.a_4) ** 2 / twog
    L_xtrk = ((h3_w - h3_target).pow(2).mean()
               + (h4_w - h4_target).pow(2).mean())

    # 5. Input tracking (anchor to mid-bound; mostly cosmetic)
    v_mid = 0.5 * (V_LO + V_HI)
    L_utrk = (((v1_w - v_mid) / (V_HI - V_LO)).pow(2).mean()
               + ((v2_w - v_mid) / (V_HI - V_LO)).pow(2).mean())

    # 6. Move suppression
    dv1 = (v1_w[:, 1:] - v1_w[:, :-1]) / (V_HI - V_LO)
    dv2 = (v2_w[:, 1:] - v2_w[:, :-1]) / (V_HI - V_LO)
    L_du = (dv1.abs() - 0.2).clamp(min=0.0).pow(2).mean() \
           + (dv2.abs() - 0.2).clamp(min=0.0).pow(2).mean()

    # 7. Input bounds (also keep h >= 0)
    L_u = (F.relu(V_LO - v1_w).pow(2) + F.relu(v1_w - V_HI).pow(2)).mean() \
          + (F.relu(V_LO - v2_w).pow(2) + F.relu(v2_w - V_HI).pow(2)).mean() \
          + F.relu(-h1_w).pow(2).mean() + F.relu(-h2_w).pow(2).mean() \
          + F.relu(-h3_w).pow(2).mean() + F.relu(-h4_w).pow(2).mean()

    # 8. NMPC behavior-cloning loss (only if u_nmpc was provided)
    # Match the PINN's action at t=1.0 (the same query the controller uses)
    # to the NMPC oracle's first-step action. Normalised by action range.
    L_nmpc = torch.tensor(0.0, device=h1_ic.device)
    if u_nmpc_batch is not None and w_nmpc > 0.0:
        t_query = torch.full_like(h1_ic, 1.0)
        _, _, _, _, v1_q, v2_q = net(
            t_query, h1_ic, h2_ic, h3_ic, h4_ic,
            h1_sp, h2_sp, v1_ic, v2_ic)
        L_nmpc = (((v1_q - u_nmpc_batch[:, 0]) / (V_HI - V_LO)).pow(2).mean()
                  + ((v2_q - u_nmpc_batch[:, 1]) / (V_HI - V_LO)).pow(2).mean())

    total = (w_ode * L_ode + w_ic * L_ic + w_ytrk * L_ytrk
              + w_xtrk * L_xtrk + w_utrk * L_utrk
              + w_du * L_du + w_u * L_u
              + w_nmpc * L_nmpc)
    components = {"L_ode": L_ode, "L_ic": L_ic, "L_ytrk": L_ytrk,
                   "L_xtrk": L_xtrk, "L_utrk": L_utrk,
                   "L_du": L_du, "L_u": L_u, "L_nmpc": L_nmpc}
    return total, components


# ============================================================================
# Two-phase Adam training loops (Kardamaki Sec 3.3 schedule)
# ============================================================================
def train_pinn_crystallization(
        net: PINN_Crystallization,
        episodes: dict,                    # {"mu0_all", "mu1_all", ..., "Tc_all"}
        hp: CrystPINNHparams,
        p: CrystParams = None,
        verbose: bool = False,
        seed: int = 0,
        ) -> dict:
    """Two-phase Adam: Phase 1 = no bounds, Phase 2 = full loss."""
    torch.manual_seed(seed)
    p = p or CrystParams()
    t_wp, t_col = make_time_grids(T_horizon=hp.T_horizon, Ts=hp.Ts)
    hist_p1, hist_p2 = [], []

    # Required keys
    keys = ["mu0_all", "mu1_all", "mu2_all", "mu3_all", "c_all",
             "cv_sp_all", "ln_sp_all", "Tc_all"]
    # Move to DEVICE once (episodes from data_gen are CPU tensors).
    arrays = [episodes[k].to(DEVICE) for k in keys]
    N_total = arrays[0].shape[0]

    # NMPC-distillation labels (used iff hp.w_nmpc > 0 AND episodes has u_nmpc).
    # Crystallization u_nmpc shape: (N_total,) — single T_c per episode.
    u_nmpc_tensor = None
    if getattr(hp, "w_nmpc", 0.0) > 0.0:
        u_nmpc_arr = episodes.get("u_nmpc")
        if u_nmpc_arr is not None:
            u_nmpc_tensor = u_nmpc_arr.to(DEVICE) if isinstance(u_nmpc_arr, torch.Tensor) \
                            else torch.from_numpy(u_nmpc_arr).to(DEVICE)

    # Importance weights for hard episodes (extreme CV/L_n setpoints)
    use_importance = getattr(hp, "importance_alpha", 0.0) > 0.0
    if use_importance:
        cv_dev = (episodes["cv_sp_all"] - 1.0).abs()
        ln_dev = (episodes["ln_sp_all"] - 15.0).abs() / 15.0
        difficulty = (cv_dev + ln_dev).cpu()
        # Weight = 1 + alpha * (rank-normalised difficulty)
        rank = torch.argsort(torch.argsort(difficulty)).float() / max(N_total - 1, 1)
        weights = 1.0 + hp.importance_alpha * rank
        weights = weights.clamp(min=1e-9)
        torch.manual_seed(seed + 1)

    def get_batch(ep_idx):
        """Returns (arrays_batch, u_nmpc_batch_or_None)."""
        if use_importance:
            idx = torch.multinomial(weights, hp.bs, replacement=True)
            idx = idx.to(arrays[0].device)
            arr_batch = [a[idx] for a in arrays]
            nmpc_batch = u_nmpc_tensor[idx] if u_nmpc_tensor is not None else None
            return arr_batch, nmpc_batch
        s = (ep_idx * hp.bs) % N_total
        e = s + hp.bs
        if e <= N_total:
            arr_batch = [a[s:e] for a in arrays]
            nmpc_batch = u_nmpc_tensor[s:e] if u_nmpc_tensor is not None else None
        else:
            arr_batch = [torch.cat([a[s:], a[:e - N_total]]) for a in arrays]
            nmpc_batch = (torch.cat([u_nmpc_tensor[s:], u_nmpc_tensor[:e - N_total]])
                          if u_nmpc_tensor is not None else None)
        return arr_batch, nmpc_batch

    # Phase 1: no bound terms
    opt = torch.optim.Adam(net.parameters(), lr=hp.lr1)
    for ep in range(1, hp.K1 + 1):
        bat, nmpc_bat = get_batch(ep - 1)
        loss, comps = composite_loss_crystallization(
            net, t_wp, t_col, *bat,
            hp.w_ode, hp.w_ic, hp.w_ytrk, hp.w_utrk,
            0.0, 0.0, 0.0, p,
            u_nmpc_batch=nmpc_bat, w_nmpc=hp.w_nmpc)
        if torch.isnan(loss):
            return {"hist_p1": hist_p1, "hist_p2": [], "nan_at": ("P1", ep)}
        opt.zero_grad(); loss.backward()
        # Gradient clipping: prevents the huge crystallization L_ode gradients
        # from blowing up the weights. max_norm=1.0 is standard for PINN training.
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
        hist_p1.append(float(loss.item()))
        if verbose and (ep == 1 or ep % 500 == 0):
            print(f"  P1 ep {ep:5d}: loss={loss.item():.4e}")

    # Phase 2: full loss
    opt = torch.optim.Adam(net.parameters(), lr=hp.lr2)
    for ep in range(1, hp.K2 + 1):
        bat, nmpc_bat = get_batch(ep - 1)
        loss, comps = composite_loss_crystallization(
            net, t_wp, t_col, *bat,
            hp.w_ode, hp.w_ic, hp.w_ytrk, hp.w_utrk,
            hp.w_du, hp.w_u, hp.w_x, p,
            u_nmpc_batch=nmpc_bat, w_nmpc=hp.w_nmpc)
        if torch.isnan(loss):
            return {"hist_p1": hist_p1, "hist_p2": hist_p2, "nan_at": ("P2", ep)}
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
        hist_p2.append(float(loss.item()))
        if verbose and (ep == 1 or ep % 500 == 0):
            print(f"  P2 ep {ep:5d}: loss={loss.item():.4e}")

    return {"hist_p1": hist_p1, "hist_p2": hist_p2, "nan_at": None}


def train_pinn_fourtank(
        net: PINN_FourTank,
        episodes: dict,
        hp: FourTankPINNHparams,
        p: FourTankParams = None,
        verbose: bool = False,
        seed: int = 0,
        ) -> dict:
    torch.manual_seed(seed)
    p = p or FourTankParams()
    t_wp, t_col = make_time_grids(T_horizon=hp.T_horizon, Ts=hp.Ts)
    hist_p1, hist_p2 = [], []
    keys = ["h1_all", "h2_all", "h3_all", "h4_all",
             "h1_sp_all", "h2_sp_all", "v1_all", "v2_all"]
    # Move to DEVICE once (episodes from data_gen are CPU tensors).
    arrays = [episodes[k].to(DEVICE) for k in keys]
    N_total = arrays[0].shape[0]

    # NMPC-distillation labels (only used if hp.w_nmpc > 0 AND episodes has u_nmpc).
    # Shape: (N_total, 2) for four-tank.
    u_nmpc_tensor = None
    if getattr(hp, "w_nmpc", 0.0) > 0.0:
        u_nmpc_arr = episodes.get("u_nmpc")
        if u_nmpc_arr is not None:
            u_nmpc_tensor = u_nmpc_arr.to(DEVICE) if isinstance(u_nmpc_arr, torch.Tensor) \
                            else torch.from_numpy(u_nmpc_arr).to(DEVICE)

    # Importance weights for hard episodes (large |h - sp|)
    use_importance = getattr(hp, "importance_alpha", 0.0) > 0.0
    if use_importance:
        d1 = (episodes["h1_all"] - episodes["h1_sp_all"]).abs()
        d2 = (episodes["h2_all"] - episodes["h2_sp_all"]).abs()
        difficulty = (d1 + d2).cpu()
        rank = torch.argsort(torch.argsort(difficulty)).float() / max(N_total - 1, 1)
        weights = 1.0 + hp.importance_alpha * rank
        weights = weights.clamp(min=1e-9)
        torch.manual_seed(seed + 1)

    def get_batch(ep_idx):
        """Returns (arrays_batch, u_nmpc_batch_or_None)."""
        if use_importance:
            idx = torch.multinomial(weights, hp.bs, replacement=True)
            idx = idx.to(arrays[0].device)
            arr_batch = [a[idx] for a in arrays]
            nmpc_batch = u_nmpc_tensor[idx] if u_nmpc_tensor is not None else None
            return arr_batch, nmpc_batch
        s = (ep_idx * hp.bs) % N_total
        e = s + hp.bs
        if e <= N_total:
            arr_batch = [a[s:e] for a in arrays]
            nmpc_batch = u_nmpc_tensor[s:e] if u_nmpc_tensor is not None else None
        else:
            arr_batch = [torch.cat([a[s:], a[:e - N_total]]) for a in arrays]
            nmpc_batch = (torch.cat([u_nmpc_tensor[s:], u_nmpc_tensor[:e - N_total]])
                          if u_nmpc_tensor is not None else None)
        return arr_batch, nmpc_batch

    opt = torch.optim.Adam(net.parameters(), lr=hp.lr1)
    for ep in range(1, hp.K1 + 1):
        bat, nmpc_bat = get_batch(ep - 1)
        loss, _ = composite_loss_fourtank(
            net, t_wp, t_col, *bat,
            hp.w_ode, hp.w_ic, hp.w_ytrk, hp.w_xtrk, hp.w_utrk,
            0.0, 0.0, p,
            u_nmpc_batch=nmpc_bat, w_nmpc=hp.w_nmpc)
        if torch.isnan(loss):
            return {"hist_p1": hist_p1, "hist_p2": [], "nan_at": ("P1", ep)}
        opt.zero_grad(); loss.backward(); opt.step()
        hist_p1.append(float(loss.item()))
        if verbose and (ep == 1 or ep % 500 == 0):
            print(f"  P1 ep {ep:5d}: loss={loss.item():.4e}")

    opt = torch.optim.Adam(net.parameters(), lr=hp.lr2)
    for ep in range(1, hp.K2 + 1):
        bat, nmpc_bat = get_batch(ep - 1)
        loss, _ = composite_loss_fourtank(
            net, t_wp, t_col, *bat,
            hp.w_ode, hp.w_ic, hp.w_ytrk, hp.w_xtrk, hp.w_utrk,
            hp.w_du, hp.w_u, p,
            u_nmpc_batch=nmpc_bat, w_nmpc=hp.w_nmpc)
        if torch.isnan(loss):
            return {"hist_p1": hist_p1, "hist_p2": hist_p2, "nan_at": ("P2", ep)}
        opt.zero_grad(); loss.backward(); opt.step()
        hist_p2.append(float(loss.item()))
        if verbose and (ep == 1 or ep % 500 == 0):
            print(f"  P2 ep {ep:5d}: loss={loss.item():.4e}")
    return {"hist_p1": hist_p1, "hist_p2": hist_p2, "nan_at": None}


# ============================================================================
# CSTR composite loss + training (LINEAR-SCALE — physics is stable here!)
# ============================================================================
def cstr_ode_residual(
        net: PINN_CSTR,
        t_col: torch.Tensor,
        C_A_ic: torch.Tensor, T_ic: torch.Tensor,
        CA_sp: torch.Tensor, Tc_ic: torch.Tensor,
        p: CSTRParams,
        ) -> torch.Tensor:
    """Returns CSTR ODE-residual squared, averaged across (B, N_col)."""
    B = C_A_ic.shape[0]
    N_col = t_col.shape[0]
    t_flat = t_col.repeat(B).requires_grad_(True)
    CA_f = C_A_ic.repeat_interleave(N_col)
    T_f  = T_ic.repeat_interleave(N_col)
    sp_f = CA_sp.repeat_interleave(N_col)
    Tc_f = Tc_ic.repeat_interleave(N_col)

    C_A_p, T_p, T_c_p = net(t_flat, CA_f, T_f, sp_f, Tc_f)
    # Time derivatives via autograd
    dCA_dt = grad(C_A_p.sum(), t_flat, create_graph=True, retain_graph=True)[0]
    dT_dt  = grad(T_p.sum(),   t_flat, create_graph=True, retain_graph=True)[0]
    # Plant ODE RHS (Eq 14-15)
    T_safe = torch.clamp(T_p, min=1e-6)
    rA = p.k0 * torch.exp(-p.EA_over_R / T_safe) * C_A_p
    rhs_CA = (p.q / p.V) * (p.C_Af - C_A_p) - rA
    rhs_T  = ((p.q / p.V) * (p.T_f - T_p)
              + (-p.deltaHr) * rA / (p.rho * p.C_p)
              + p.UA * (T_c_p - T_p) / (p.rho * p.C_p * p.V))
    # Normalised residuals (so different scales don't dominate)
    res_CA = (dCA_dt - rhs_CA) / (C_A_HI - C_A_LO)
    res_T  = (dT_dt  - rhs_T)  / (CSTR_T_HI - CSTR_T_LO)
    return (res_CA.pow(2).mean() + res_T.pow(2).mean())


def composite_loss_cstr(
        net: PINN_CSTR,
        t_wp: torch.Tensor, t_col: torch.Tensor,
        C_A_ic, T_ic, CA_sp, Tc_ic,
        w_ode, w_ic, w_ytrk, w_utrk, w_du, w_u, w_x,
        p: CSTRParams,
        u_nmpc_batch: torch.Tensor | None = None,
        w_nmpc: float = 0.0) -> tuple[torch.Tensor, dict]:
    """Full composite loss for the CSTR PINN."""
    B = C_A_ic.shape[0]
    N_WP = t_wp.shape[0]
    _zero = torch.tensor(0.0, device=C_A_ic.device)

    # 1. ODE residual (only if w_ode > 0)
    if w_ode > 0:
        L_ode = cstr_ode_residual(net, t_col, C_A_ic, T_ic, CA_sp, Tc_ic, p)
    else:
        L_ode = _zero

    # Multi-waypoint forward (needed for L_ic, L_ytrk, etc.)
    need_waypoints = any(w > 0 for w in (w_ic, w_ytrk, w_utrk, w_du, w_u, w_x))
    if need_waypoints:
        t_flat = t_wp.repeat(B)
        CA_f = C_A_ic.repeat_interleave(N_WP)
        T_f  = T_ic.repeat_interleave(N_WP)
        sp_f = CA_sp.repeat_interleave(N_WP)
        Tc_f = Tc_ic.repeat_interleave(N_WP)
        CA_p, T_p, Tc_p = net(t_flat, CA_f, T_f, sp_f, Tc_f)
        CA_w = CA_p.view(B, N_WP)
        T_w  = T_p.view(B, N_WP)
        Tc_w = Tc_p.view(B, N_WP)

        # 2. IC (state at t=0 matches IC)
        L_ic = (((CA_w[:, 0] - C_A_ic) / (C_A_HI - C_A_LO)).pow(2).mean()
                + ((T_w[:, 0] - T_ic) / (CSTR_T_HI - CSTR_T_LO)).pow(2).mean())

        # 3. Output tracking (C_A -> CA_sp)
        L_ytrk = (((CA_w - CA_sp.unsqueeze(1)) / (C_A_HI - C_A_LO))
                   .pow(2).mean())

        # 4. Input tracking (T_c -> mid-bound)
        Tc_mid = 0.5 * (CSTR_TC_LO + CSTR_TC_HI)
        L_utrk = (((Tc_w - Tc_mid) / (CSTR_TC_HI - CSTR_TC_LO))
                   .pow(2).mean())

        # 5. Move suppression
        dTc = (Tc_w[:, 1:] - Tc_w[:, :-1]) / (CSTR_TC_HI - CSTR_TC_LO)
        L_du = (dTc.abs() - 0.1).clamp(min=0.0).pow(2).mean()

        # 6. Input bounds
        L_u = (F.relu(CSTR_TC_LO - Tc_w).pow(2)
               + F.relu(Tc_w - CSTR_TC_HI).pow(2)).mean()

        # 7. State bounds (C_A >= 0; T in normal range)
        L_x = F.relu(-CA_w).pow(2).mean()
    else:
        L_ic = L_ytrk = L_utrk = L_du = L_u = L_x = _zero

    # 8. NMPC behavior-cloning (matches PINN's T_c at t=1.0 to oracle u_NMPC)
    L_nmpc = torch.tensor(0.0, device=C_A_ic.device)
    if u_nmpc_batch is not None and w_nmpc > 0:
        t_query = torch.full_like(C_A_ic, 1.0)
        _, _, Tc_q = net(t_query, C_A_ic, T_ic, CA_sp, Tc_ic)
        L_nmpc = ((Tc_q - u_nmpc_batch) / (CSTR_TC_HI - CSTR_TC_LO)).pow(2).mean()

    total = (w_ode * L_ode + w_ic * L_ic + w_ytrk * L_ytrk
              + w_utrk * L_utrk + w_du * L_du + w_u * L_u + w_x * L_x
              + w_nmpc * L_nmpc)
    components = {"L_ode": L_ode, "L_ic": L_ic, "L_ytrk": L_ytrk,
                   "L_utrk": L_utrk, "L_du": L_du, "L_u": L_u, "L_x": L_x,
                   "L_nmpc": L_nmpc}
    return total, components


def train_pinn_cstr(
        net: PINN_CSTR,
        episodes: dict,
        hp: CSTRPINNHparams,
        p: CSTRParams = None,
        verbose: bool = False,
        seed: int = 0,
        ) -> dict:
    """Two-phase Adam training for the CSTR PINN."""
    torch.manual_seed(seed)
    p = p or CSTRParams()
    t_wp, t_col = make_time_grids(T_horizon=hp.T_horizon, Ts=hp.Ts)
    hist_p1, hist_p2 = [], []
    keys = ["C_A_all", "T_all", "CA_sp_all", "T_c_ic_all"]
    arrays = [episodes[k].to(DEVICE) for k in keys]
    N_total = arrays[0].shape[0]

    # NMPC labels (only if hp.w_nmpc > 0)
    u_nmpc_tensor = None
    if getattr(hp, "w_nmpc", 0.0) > 0.0:
        u_nmpc_arr = episodes.get("u_nmpc")
        if u_nmpc_arr is not None:
            u_nmpc_tensor = u_nmpc_arr.to(DEVICE) if isinstance(u_nmpc_arr, torch.Tensor) \
                            else torch.from_numpy(u_nmpc_arr).to(DEVICE)

    def get_batch(ep_idx):
        s = (ep_idx * hp.bs) % N_total
        e = s + hp.bs
        if e <= N_total:
            arr_batch = [a[s:e] for a in arrays]
            nmpc_batch = u_nmpc_tensor[s:e] if u_nmpc_tensor is not None else None
        else:
            arr_batch = [torch.cat([a[s:], a[:e - N_total]]) for a in arrays]
            nmpc_batch = (torch.cat([u_nmpc_tensor[s:], u_nmpc_tensor[:e - N_total]])
                          if u_nmpc_tensor is not None else None)
        return arr_batch, nmpc_batch

    # Phase 1: no bound terms (w_du, w_u, w_x = 0)
    opt = torch.optim.Adam(net.parameters(), lr=hp.lr1)
    for ep in range(1, hp.K1 + 1):
        bat, nmpc_bat = get_batch(ep - 1)
        loss, _ = composite_loss_cstr(
            net, t_wp, t_col, *bat,
            hp.w_ode, hp.w_ic, hp.w_ytrk, hp.w_utrk,
            0.0, 0.0, 0.0, p,
            u_nmpc_batch=nmpc_bat, w_nmpc=hp.w_nmpc)
        if torch.isnan(loss):
            return {"hist_p1": hist_p1, "hist_p2": [], "nan_at": ("P1", ep)}
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
        hist_p1.append(float(loss.item()))
        if verbose and (ep == 1 or ep % 500 == 0):
            print(f"  P1 ep {ep:5d}: loss={loss.item():.4e}")

    # Phase 2: full loss
    opt = torch.optim.Adam(net.parameters(), lr=hp.lr2)
    for ep in range(1, hp.K2 + 1):
        bat, nmpc_bat = get_batch(ep - 1)
        loss, _ = composite_loss_cstr(
            net, t_wp, t_col, *bat,
            hp.w_ode, hp.w_ic, hp.w_ytrk, hp.w_utrk,
            hp.w_du, hp.w_u, hp.w_x, p,
            u_nmpc_batch=nmpc_bat, w_nmpc=hp.w_nmpc)
        if torch.isnan(loss):
            return {"hist_p1": hist_p1, "hist_p2": hist_p2, "nan_at": ("P2", ep)}
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
        hist_p2.append(float(loss.item()))
        if verbose and (ep == 1 or ep % 500 == 0):
            print(f"  P2 ep {ep:5d}: loss={loss.item():.4e}")

    return {"hist_p1": hist_p1, "hist_p2": hist_p2, "nan_at": None}
