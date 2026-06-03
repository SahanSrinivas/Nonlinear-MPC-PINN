"""PINN-MPC for the K2SO4 crystallization reactor (PC-Gym Case Study 3).

Adapts the Kardamaki 2026 PINN-MPC architecture (J. Process Control 158)
to a 5-state plant with 1 input and 2 controlled outputs (CV, L_n).

Input  to net: [t, mu_0_IC, mu_1_IC, mu_2_IC, mu_3_IC, c_IC, CV_sp, Ln_sp, T_c_IC]
Output of net: [mu_0(t), mu_1(t), mu_2(t), mu_3(t), c(t), T_c(t)]

Architecture: 3 hidden layers x 64 units, Tanh  (Kardamaki default for SISO).
Composite loss matches Kardamaki Eq. (17):
    L = w_ode*L_ode + w_ic*L_IC + w_ytrk*L_ytrk + w_utrk*L_utrk
        + w_du*L_du + w_u*L_u + w_x*L_x

State scaling is critical because mu_0..mu_3 span many orders of magnitude:
we apply log-scale normalisation on inputs and outputs (mu_i_norm = log10(mu_i)
clamped to a reasonable range).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Normalisation helpers (mu_i span many orders of magnitude in linear scale)
# ============================================================================
# Bounds were tuned to comfortably bracket the trajectory ranges observed in
# the NMPC smoke run. They are used ONLY for network input/output scaling.
MU_0_LOG_LO, MU_0_LOG_HI = -2.0,  10.0   # log10(mu_0)  : ~ [0.01, 1e10]
MU_1_LOG_LO, MU_1_LOG_HI = -2.0,  10.0
MU_2_LOG_LO, MU_2_LOG_HI = -2.0,  10.0
MU_3_LOG_LO, MU_3_LOG_HI = -2.0,  10.0
C_LO,        C_HI        =  0.0,  1.0    # mol/L solute concentration
T_C_LO,      T_C_HI      =  25.0, 50.0   # degC (matches NMPC bounds)
CV_LO, CV_HI = 0.0, 3.0
LN_LO, LN_HI = 0.0, 30.0


def _log_norm(v: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """log10-then-normalize to roughly [0, 1]."""
    lv = torch.log10(v.clamp(min=1e-30))
    return (lv - lo) / (hi - lo)


def _log_denorm(s: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Inverse of _log_norm."""
    lv = s * (hi - lo) + lo
    return torch.pow(10.0, lv)


# ============================================================================
# Architecture: feedforward MLP with tanh + final linear
# ============================================================================
class PINN_Crystallization(nn.Module):
    """PINN-MPC controller for the crystallization reactor.

    Inputs (9):  t, mu_0_IC, mu_1_IC, mu_2_IC, mu_3_IC, c_IC, CV_sp, Ln_sp, T_c_IC
    Outputs (6): mu_0(t), mu_1(t), mu_2(t), mu_3(t), c(t), T_c(t)

    All mu's are log-normalised; c, T_c, CV_sp, Ln_sp use linear normalisation.
    """

    INPUT_DIM = 9
    OUTPUT_DIM = 6

    def __init__(self, hidden_layers: List[int] = [64, 64, 64]):
        super().__init__()
        layers = []
        in_dim = self.INPUT_DIM
        for w in hidden_layers:
            layers += [nn.Linear(in_dim, w), nn.Tanh()]
            in_dim = w
        layers += [nn.Linear(in_dim, self.OUTPUT_DIM)]
        self.net = nn.Sequential(*layers)
        # Bounds
        self.T_c_min, self.T_c_max = T_C_LO, T_C_HI

    @staticmethod
    def normalize_inputs(t, mu0_ic, mu1_ic, mu2_ic, mu3_ic, c_ic,
                           cv_sp, ln_sp, Tc_ic, t_norm: float = 30.0):
        """Returns (B, 9) input tensor in normalised space."""
        return torch.stack((
            t / t_norm,
            _log_norm(mu0_ic, MU_0_LOG_LO, MU_0_LOG_HI),
            _log_norm(mu1_ic, MU_1_LOG_LO, MU_1_LOG_HI),
            _log_norm(mu2_ic, MU_2_LOG_LO, MU_2_LOG_HI),
            _log_norm(mu3_ic, MU_3_LOG_LO, MU_3_LOG_HI),
            (c_ic - C_LO) / (C_HI - C_LO),
            (cv_sp - CV_LO) / (CV_HI - CV_LO),
            (ln_sp - LN_LO) / (LN_HI - LN_LO),
            (Tc_ic - T_C_LO) / (T_C_HI - T_C_LO),
        ), dim=-1)

    def forward(self, t, mu0_ic, mu1_ic, mu2_ic, mu3_ic, c_ic,
                cv_sp, ln_sp, Tc_ic):
        """Returns (mu_0, mu_1, mu_2, mu_3, c, T_c), each in physical units."""
        inp = self.normalize_inputs(t, mu0_ic, mu1_ic, mu2_ic, mu3_ic, c_ic,
                                       cv_sp, ln_sp, Tc_ic)
        out = self.net(inp)  # (B, 6) in normalised space
        # Denormalise to physical units
        mu0_pred = _log_denorm(out[..., 0], MU_0_LOG_LO, MU_0_LOG_HI)
        mu1_pred = _log_denorm(out[..., 1], MU_1_LOG_LO, MU_1_LOG_HI)
        mu2_pred = _log_denorm(out[..., 2], MU_2_LOG_LO, MU_2_LOG_HI)
        mu3_pred = _log_denorm(out[..., 3], MU_3_LOG_LO, MU_3_LOG_HI)
        c_pred   = out[..., 4] * (C_HI - C_LO) + C_LO
        Tc_pred  = out[..., 5] * (T_C_HI - T_C_LO) + T_C_LO
        return mu0_pred, mu1_pred, mu2_pred, mu3_pred, c_pred, Tc_pred


# ============================================================================
# Hparams
# ============================================================================
@dataclass
class CrystPINNHparams:
    """Loss weights and training settings for the crystallization PINN."""
    # Loss weights (literature-cited starting points; LLM-AutoOpt tunes these)
    w_ode:   float = 100.0
    w_ic:    float = 10.0
    w_ytrk:  float = 10.0    # output (CV, L_n) tracking
    w_utrk:  float = 1.0     # input (T_c) tracking
    w_du:    float = 1.0     # move suppression
    w_u:     float = 100.0   # input bound violation
    w_x:     float = 10.0    # state bound violation
    # Optimizer
    lr1:     float = 1e-3
    lr2:     float = 2e-4
    K1:      int   = 10000
    K2:      int   = 10000
    bs:      int   = 64
    # Time discretisation (30 hr episode, 30 steps)
    T_horizon: float = 30.0   # hours
    Ts:        float = 1.0    # controller dt (hours)
    # Importance sampling on hard episodes (extreme setpoints).
    # alpha=0 -> uniform; alpha=1 -> hard episodes (top quantile of |CV-1|+|Ln-15|/15)
    # weighted up to (1+alpha)x relative to easy episodes.
    importance_alpha: float = 0.0
    importance_quantile: float = 0.5


# Self-test (architecture only - no training here)
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    net = PINN_Crystallization().to(DEVICE)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"PINN_Crystallization: {n_params} trainable params, "
          f"input dim={net.INPUT_DIM}, output dim={net.OUTPUT_DIM}")
    # Single forward pass smoke test
    t   = torch.tensor([1.0,  5.0, 15.0], device=DEVICE)
    mu0 = torch.tensor([1.0,  1.0,  1.0], device=DEVICE)
    mu1 = torch.tensor([15.0, 15.0, 15.0], device=DEVICE)
    mu2 = torch.tensor([1125.0]*3, device=DEVICE)
    mu3 = torch.tensor([3375.0]*3, device=DEVICE)
    c   = torch.tensor([0.30,0.30,0.30], device=DEVICE)
    cvsp= torch.tensor([1.0, 1.0, 1.0], device=DEVICE)
    lnsp= torch.tensor([15.0,15.0,15.0], device=DEVICE)
    Tc0 = torch.tensor([32.0,32.0,32.0], device=DEVICE)
    out = net(t, mu0, mu1, mu2, mu3, c, cvsp, lnsp, Tc0)
    print(f"Forward pass OK. Output shapes: " +
          ", ".join(f"{o.shape}" for o in out))
    print(f"  mu_0(t)={out[0].cpu().detach().numpy()}")
    print(f"  T_c(t) ={out[5].cpu().detach().numpy()}")
