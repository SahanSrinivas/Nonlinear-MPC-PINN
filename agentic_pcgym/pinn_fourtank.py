"""PINN-MPC for the four-tank MIMO system (PC-Gym Case Study 4).

Adapts Kardamaki 2026 Section 4.2 (MIMO PINN-MPC for the quadruple-tank) to
PC-Gym's four-tank. 4 states, 2 inputs, 2 controlled outputs.

Input  to net (9):  t, h_1_IC, h_2_IC, h_3_IC, h_4_IC, h1_sp, h2_sp, v_1_IC, v_2_IC
Output of net (6):  h_1(t), h_2(t), h_3(t), h_4(t), v_1(t), v_2(t)

Architecture matches Kardamaki Sec 4.2: 3 hidden layers x 64 units, Tanh.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Bounds (from NMPC analysis - in `nmpc_fourtank.FourTankBounds`)
V_LO, V_HI = 0.0, 15.0
H_LO, H_HI = 0.0, 1.5


class PINN_FourTank(nn.Module):
    """MIMO PINN-MPC controller for the four-tank plant.

    Same structural pattern as Kardamaki Sec 4.2: a feedforward MLP that
    predicts the full state + input trajectory, trained via the composite
    physics-informed loss.

    Inputs (9):  t, h_1_IC, h_2_IC, h_3_IC, h_4_IC, h1_sp, h2_sp, v1_IC, v2_IC
    Outputs (6): h_1(t), h_2(t), h_3(t), h_4(t), v_1(t), v_2(t)

    All states / actions normalised to [0,1] via the bounds above.
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
        # Plant constants (frozen at PINN training; LLM-AutoOpt tunes loss
        # weights, not these).
        self.v_min, self.v_max = V_LO, V_HI
        self.h_min, self.h_max = H_LO, H_HI

    @staticmethod
    def normalize_inputs(t, h1_ic, h2_ic, h3_ic, h4_ic,
                          h1_sp, h2_sp, v1_ic, v2_ic,
                          t_norm: float = 1000.0):
        return torch.stack((
            t / t_norm,
            (h1_ic - H_LO) / (H_HI - H_LO),
            (h2_ic - H_LO) / (H_HI - H_LO),
            (h3_ic - H_LO) / (H_HI - H_LO),
            (h4_ic - H_LO) / (H_HI - H_LO),
            (h1_sp - H_LO) / (H_HI - H_LO),
            (h2_sp - H_LO) / (H_HI - H_LO),
            (v1_ic - V_LO) / (V_HI - V_LO),
            (v2_ic - V_LO) / (V_HI - V_LO),
        ), dim=-1)

    def forward(self, t, h1_ic, h2_ic, h3_ic, h4_ic,
                h1_sp, h2_sp, v1_ic, v2_ic):
        """Returns (h_1, h_2, h_3, h_4, v_1, v_2) in physical units."""
        inp = self.normalize_inputs(t, h1_ic, h2_ic, h3_ic, h4_ic,
                                       h1_sp, h2_sp, v1_ic, v2_ic)
        out = self.net(inp)
        h1_p = out[..., 0] * (H_HI - H_LO) + H_LO
        h2_p = out[..., 1] * (H_HI - H_LO) + H_LO
        h3_p = out[..., 2] * (H_HI - H_LO) + H_LO
        h4_p = out[..., 3] * (H_HI - H_LO) + H_LO
        v1_p = out[..., 4] * (V_HI - V_LO) + V_LO
        v2_p = out[..., 5] * (V_HI - V_LO) + V_LO
        return h1_p, h2_p, h3_p, h4_p, v1_p, v2_p


@dataclass
class FourTankPINNHparams:
    """Loss weights and training settings for the four-tank PINN."""
    w_ode:   float = 100.0
    w_ic:    float = 10.0
    # Note: Kardamaki Sec 4.2 adds a w_xtrk for upper-tank steady-state
    # tracking. We include it here too.
    w_ytrk:  float = 10.0    # h_1, h_2 tracking
    w_xtrk:  float = 1.0     # h_3, h_4 (non-controlled state) steady-state tracking
    w_utrk:  float = 1.0     # v_1, v_2 tracking
    w_du:    float = 1.0     # move suppression
    w_u:     float = 100.0   # input bounds
    # NMPC behavior-cloning weight: matches PINN(t=1.0, x_IC, sp, u_prev) to
    # NMPC oracle action u_NMPC(x_IC, sp). Active only when episodes contain
    # "u_nmpc" tensor (i.e., when sampled with query_nmpc=True). Default 0
    # preserves backward compatibility for physics-only training.
    w_nmpc:  float = 0.0
    # Optimizer
    lr1:     float = 1e-3
    lr2:     float = 2e-4
    K1:      int   = 10000
    K2:      int   = 10000
    bs:      int   = 64
    # Time discretisation (1000 s episode, 60 steps)
    T_horizon: float = 1000.0   # seconds
    Ts:        float = 1000.0 / 60.0   # ~16.67 s
    # Importance sampling on hard episodes (large |x0 - sp|).
    # alpha=0 -> uniform; alpha=1 -> hard episodes weighted up to 2x.
    importance_alpha: float = 0.0
    importance_quantile: float = 0.5


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    net = PINN_FourTank().to(DEVICE)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"PINN_FourTank: {n_params} trainable params, "
          f"input dim={net.INPUT_DIM}, output dim={net.OUTPUT_DIM}")
    # Single forward pass smoke test
    t   = torch.tensor([10.0, 100.0, 500.0], device=DEVICE)
    h1  = torch.tensor([0.4, 0.4, 0.4], device=DEVICE)
    h2  = torch.tensor([0.2, 0.2, 0.2], device=DEVICE)
    h3  = torch.tensor([0.2, 0.2, 0.2], device=DEVICE)
    h4  = torch.tensor([0.2, 0.2, 0.2], device=DEVICE)
    sp1 = torch.tensor([0.5, 0.5, 0.5], device=DEVICE)
    sp2 = torch.tensor([0.3, 0.3, 0.3], device=DEVICE)
    v1  = torch.tensor([5.0, 5.0, 5.0], device=DEVICE)
    v2  = torch.tensor([5.0, 5.0, 5.0], device=DEVICE)
    out = net(t, h1, h2, h3, h4, sp1, sp2, v1, v2)
    print(f"Forward pass OK. Outputs:")
    print(f"  h_1(t) = {out[0].cpu().detach().numpy()}")
    print(f"  v_1(t) = {out[4].cpu().detach().numpy()}")
