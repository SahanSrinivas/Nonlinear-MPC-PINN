"""PINN-MPC for the CSTR (PC-Gym Case Study 1).

Adapts the Kardamaki 2026 PINN-MPC architecture (J. Process Control 158)
to a simple 2-state plant with 1 input and 1 controlled output (C_A).

Input  to net: [t, C_A_IC, T_IC, CA_sp, T_c_IC]   (5 inputs)
Output of net: [C_A(t), T(t), T_c(t)]              (3 outputs)

Architecture: 3 hidden layers x 64 units, Tanh (Kardamaki default for SISO).
This is a SIMPLE problem (CSTR has 2 states, no log scaling needed) — small
network should suffice.

Composite loss matches Kardamaki Eq. (17):
    L = w_ode*L_ode + w_ic*L_IC + w_ytrk*L_ytrk + w_utrk*L_utrk
        + w_du*L_du + w_u*L_u + w_x*L_x  + w_nmpc*L_nmpc

NOTE: CSTR uses LINEAR normalisation (no log10), so physics losses are
stable here — unlike crystallization. We CAN enable physics losses for CSTR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Bounds (input/output normalisation)
# ============================================================================
T_C_LO, T_C_HI = 295.0, 302.0   # Bloor §4.1.1
C_A_LO, C_A_HI = 0.0,   1.0     # C_A operating range (paper)
T_LO,   T_HI   = 300.0, 400.0   # T operating range (typical CSTR)


# ============================================================================
# Architecture: feedforward MLP with tanh + final linear
# ============================================================================
class PINN_CSTR(nn.Module):
    """PINN-MPC controller for the CSTR plant.

    Inputs (5):  t, C_A_IC, T_IC, CA_sp, T_c_IC
    Outputs (3): C_A(t), T(t), T_c(t)
    """

    INPUT_DIM = 5
    OUTPUT_DIM = 3

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
    def normalize_inputs(t, C_A_ic, T_ic, CA_sp, T_c_ic,
                          t_norm: float = 25.0):
        """Returns (B, 5) input tensor in normalised space."""
        return torch.stack((
            t / t_norm,
            (C_A_ic - C_A_LO) / (C_A_HI - C_A_LO),
            (T_ic - T_LO) / (T_HI - T_LO),
            (CA_sp - C_A_LO) / (C_A_HI - C_A_LO),
            (T_c_ic - T_C_LO) / (T_C_HI - T_C_LO),
        ), dim=-1)

    def forward(self, t, C_A_ic, T_ic, CA_sp, T_c_ic):
        """Returns (C_A, T, T_c), each in physical units."""
        inp = self.normalize_inputs(t, C_A_ic, T_ic, CA_sp, T_c_ic)
        out = self.net(inp)  # (B, 3) in normalised space
        # Denormalise to physical units (linear, no log)
        C_A_pred = out[..., 0] * (C_A_HI - C_A_LO) + C_A_LO
        T_pred   = out[..., 1] * (T_HI - T_LO) + T_LO
        T_c_pred = out[..., 2] * (T_C_HI - T_C_LO) + T_C_LO
        return C_A_pred, T_pred, T_c_pred


# ============================================================================
# Hyperparameters
# ============================================================================
@dataclass
class CSTRPINNHparams:
    """Loss weights and training settings for the CSTR PINN."""
    # Loss weights (literature-cited starting points)
    w_ode:   float = 100.0
    w_ic:    float = 10.0
    w_ytrk:  float = 10.0    # C_A tracking
    w_utrk:  float = 1.0     # T_c tracking
    w_du:    float = 1.0     # move suppression
    w_u:     float = 100.0   # input bound violation
    w_x:     float = 10.0    # state bound violation
    # NMPC behavior-cloning weight
    w_nmpc:  float = 0.0
    # Optimizer
    lr1:     float = 1e-3
    lr2:     float = 2e-4
    K1:      int   = 10000
    K2:      int   = 10000
    bs:      int   = 64
    # Time discretisation
    T_horizon: float = 25.0      # minutes (paper §4.3.1)
    Ts:        float = 25.0 / 60.0   # ~0.4167 min per controller step
    # Importance sampling on hard episodes (large |C_A - sp|)
    importance_alpha: float = 0.0
    importance_quantile: float = 0.5
