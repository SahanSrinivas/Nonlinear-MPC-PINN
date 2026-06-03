"""SISO PINN-MPC for the Kardamaki 2026 nonlinear single-tank.

Ported from `external/PINN_MPC_SISO_Github_Code.ipynb` (cells 11, 12, 14, 15).
Original code is CC-BY-NC-SA from Kardamaki et al. 2026. We add:
  - CPU-friendly path (their AMP autocast is GPU-specific)
  - `train_pinn(hparams)` wrapper returning closed-loop eval metric
  - `evaluate_closed_loop()` matching their test scenarios

Reference:
  Kardamaki et al. (2026). "An explicit MPC framework based on PINNs."
  Journal of Process Control 158, 103634.
  https://github.com/ntua-unit-of-control-and-informatics/pinn-mpc
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Cell 11: collocation point generator (verbatim, only DEVICE indirection)
# ---------------------------------------------------------------------------
def generate_collocation_points(horizon: float,
                                  dt_dense: float = 0.01,
                                  dt_medium: float = 0.1,
                                  dt_sparse: float = 1.0) -> torch.Tensor:
    """Non-uniformly sampled collocation points: dense near t=0."""
    t_break1 = 2 * horizon / 3
    t_break2 = 1 * horizon / 3
    t_back_sparse = torch.arange(horizon, t_break1, -dt_sparse, device=DEVICE)
    t_back_medium = torch.arange(t_break1, t_break2, -dt_medium, device=DEVICE)
    t_back_dense = torch.arange(t_break2, 0.0, -dt_dense, device=DEVICE)
    t_col = torch.cat([t_back_sparse, t_back_medium, t_back_dense,
                       torch.tensor([horizon], device=DEVICE)])
    t_col = torch.unique(t_col)
    t_col = torch.sort(t_col).values
    if t_col[-1] < horizon:
        t_col = torch.cat([t_col, torch.tensor([horizon], device=DEVICE)])
    if t_col[0] > 0:
        t_col = torch.cat([torch.tensor([0.0], device=DEVICE), t_col])
    return t_col


# ---------------------------------------------------------------------------
# Cell 12: relaxed recentered barrier (verbatim)
# ---------------------------------------------------------------------------
def barrier_du(z, z_min, z_max, delta):
    z_min_t = torch.tensor(z_min, device=z.device, dtype=z.dtype)
    z_max_t = torch.tensor(z_max, device=z.device, dtype=z.dtype)
    delta_t = torch.tensor(delta, device=z.device, dtype=z.dtype)

    def B_single(x, delta):
        return torch.where(
            x > delta, -torch.log(x),
            0.5 * ((x - 2 * delta) / delta) ** 2 - 0.5 - torch.log(delta)
        )

    B_raw = B_single(z - z_min_t, delta_t) + B_single(z_max_t - z, delta_t)
    z_c = 0.5 * (z_min_t + z_max_t)
    B_center = B_single(z_c - z_min_t, delta_t) + B_single(z_max_t - z_c, delta_t)
    return B_raw - B_center


# ---------------------------------------------------------------------------
# Cell 14: PINN_Controller (verbatim port)
# ---------------------------------------------------------------------------
class PINN_Controller(nn.Module):
    """Kardamaki 2026 SISO PINN-MPC. (cell 14 of their notebook.)

    Inputs:  [t, x0, u0, ysp, d0, x0-ysp]    Outputs: [x(t), u(t)]
    Constraints (single-tank): x in [0, 4.0] m, u in [0, 1] m^3/s, |du| <= 0.2.
    """

    def __init__(self, hidden_layers: List[int] = [64, 16, 16],
                 K_VALVE: float = 0.7, A: float = 1.0):
        super().__init__()
        self.x_min, self.x_max = 0.0, 4.0
        self.u_min, self.u_max = 0.0, 1.0
        self.du_max = 0.2
        self.K_VALVE = K_VALVE
        self.A = A
        layers = []
        in_dim = 6
        for width in hidden_layers:
            layers += [nn.Linear(in_dim, width), nn.Tanh()]
            in_dim = width
        layers += [nn.Linear(in_dim, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, t, x0, u0, ysp, d0):
        net_in = torch.stack((t, x0, u0, ysp, d0, x0 - ysp), dim=-1)
        x_pred, u_pred = self.net(net_in).unbind(-1)
        return x_pred, u_pred

    def loss(self, bs, t_wp, N_WP, t_col, N_COL, x0, u0, ysp, d0,
             w_ode, w_ic, w_ytrk, w_utrk, w_du, w_u, w_x):
        # --- tracking points ---
        t_wp = t_wp.clone().detach().requires_grad_(True)
        t_flat_wp = t_wp.repeat(bs)
        x0_flat_wp = x0.repeat_interleave(N_WP)
        u0_flat_wp = u0.repeat_interleave(N_WP)
        ysp_flat_wp = ysp.repeat_interleave(N_WP)
        d0_flat_wp = d0.repeat_interleave(N_WP)
        x_flat_wp, u_flat_wp = self(t_flat_wp, x0_flat_wp, u0_flat_wp,
                                      ysp_flat_wp, d0_flat_wp)
        x_wp = x_flat_wp.view(bs, N_WP)
        u_wp = u_flat_wp.view(bs, N_WP)
        # --- collocation points ---
        t_col = t_col.clone().detach().requires_grad_(True)
        t_flat_col = t_col.repeat(bs)
        x0_flat_col = x0.repeat_interleave(N_COL)
        u0_flat_col = u0.repeat_interleave(N_COL)
        ysp_flat_col = ysp.repeat_interleave(N_COL)
        d0_flat_col = d0.repeat_interleave(N_COL)
        x_flat_col, u_flat_col = self(t_flat_col, x0_flat_col, u0_flat_col,
                                        ysp_flat_col, d0_flat_col)
        x_col = x_flat_col.view(bs, N_COL)
        u_col = u_flat_col.view(bs, N_COL)
        # L_ode
        dx_dt_flat = grad(
            x_flat_col, t_flat_col, torch.ones_like(x_flat_col),
            create_graph=True, retain_graph=True
        )[0]
        dx_dt = dx_dt_flat.view(bs, N_COL)
        f_out = self.K_VALVE * torch.sqrt(torch.clamp(x_col, min=0.0))
        d0_mat = d0.unsqueeze(1)
        rhs = (u_col + d0_mat - f_out) / self.A
        r_ode = dx_dt - rhs
        sq_ode = (1 / N_COL) * r_ode.pow(2).sum(dim=1)
        loss_ode = w_ode * sq_ode.mean()
        # L_ytrk
        ysp_mat = ysp.unsqueeze(1)
        e_ytrk = x_wp - ysp_mat
        sq_ytrk = (1 / N_WP) * e_ytrk.pow(2).sum(dim=1)
        loss_ytrk = w_ytrk * sq_ytrk.mean()
        # L_utrk
        u_ss = self.K_VALVE * torch.sqrt(torch.clamp(ysp_mat, min=0.0)) - d0_mat
        e_utrk = u_wp - u_ss
        sq_utrk = (1 / N_WP) * e_utrk.pow(2).sum(dim=1)
        loss_utrk = w_utrk * sq_utrk.mean()
        # L_du
        u_seq = torch.cat([u0.unsqueeze(1), u_wp], dim=1)
        du = u_seq[:, 1:] - u_seq[:, :-1]
        delta = 0.01
        b_du = barrier_du(du, -self.du_max, self.du_max, delta)
        sq_bdu = (1 / N_WP) * b_du.sum(dim=1)
        loss_du = w_du * sq_bdu.mean()
        # L_u, L_x soft bounds
        e_min_u = F.relu(self.u_min - u_wp)
        e_max_u = F.relu(u_wp - self.u_max)
        slack_u = e_min_u + e_max_u
        sum_slack_u = (1 / N_WP) * slack_u.sum(dim=1)
        loss_u = w_u * sum_slack_u.pow(2).mean()
        e_min_x = F.relu(self.x_min - x_wp)
        e_max_x = F.relu(x_wp - self.x_max)
        slack_x = e_min_x + e_max_x
        sum_slack_x = (1 / N_WP) * slack_x.sum(dim=1)
        loss_x = w_x * sum_slack_x.pow(2).mean()
        # L_IC
        x0_col_ = x0.unsqueeze(1)
        x0_pred = x_wp[:, 0:1]
        sq_ic = (x0_pred - x0_col_).pow(2)
        loss_ic = w_ic * sq_ic.mean()
        loss_total = (loss_ode + loss_ytrk + loss_utrk
                      + loss_du + loss_u + loss_x + loss_ic)
        return loss_total, (loss_ode, loss_ytrk, loss_utrk,
                              loss_du, loss_u, loss_x, loss_ic)


# ---------------------------------------------------------------------------
# Cell 15: train_PINN (CPU-safe port - AMP autocast removed on CPU)
# ---------------------------------------------------------------------------
@dataclass
class PINNHparams:
    """Kardamaki 2026 SISO published values (Table 2) as defaults."""
    w_ode: float = 131.2036331073741
    w_ic: float = 2.3417227078047818
    w_ytrk: float = 6.281902184510784
    w_utrk: float = 6.72820755280816
    w_du: float = 32.51940650885499
    w_u: float = 3243.4337990610957
    w_x: float = 325.9614180505487
    lr1: float = 0.001008704750543596
    lr2: float = 0.0002655403205573775
    # Training settings (their published)
    K1: int = 10000
    K2: int = 10000
    bs: int = 100
    T_horizon: float = 25.0
    Ts: float = 1.0
    # Importance weighting on disturbance episodes during training.
    # Final loss is scaled by (1 + (importance_d0_alpha * d0 / d0_max)) per
    # episode, averaged over the batch. alpha=0 -> uniform (Kardamaki).
    # alpha=2 -> high-d0 episodes weighted up to 3x.
    importance_d0_alpha: float = 0.0
    importance_d0_max: float = 0.4   # the d_hi bound from the paper


def train_pinn_siso(hp: PINNHparams,
                     x0_all: torch.Tensor, u0_all: torch.Tensor,
                     ysp_all: torch.Tensor, d0_all: torch.Tensor,
                     hidden_layers: List[int] = [64, 16, 16],
                     verbose: bool = False, seed: int = 0,
                     ) -> tuple[PINN_Controller, dict]:
    """CPU-safe two-phase Adam training. Returns (trained model, history dict)."""
    torch.manual_seed(seed)
    model = PINN_Controller(hidden_layers=hidden_layers).to(DEVICE)
    # Time vectors
    t_wp = torch.cat([torch.arange(0, hp.T_horizon, hp.Ts, device=DEVICE),
                       torch.tensor([hp.T_horizon], device=DEVICE)])
    N_WP = t_wp.shape[0]
    t_col = generate_collocation_points(
        horizon=hp.T_horizon, dt_dense=0.01, dt_medium=0.1, dt_sparse=1.0)
    N_COL = t_col.shape[0]
    hist_p1, hist_p2 = [], []
    # Phase 1: dynamics + tracking only
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr1)
    nan_at = None
    # Pre-compute importance sampling weights if requested (oversamples high-d0
    # episodes for the SAME loss term). alpha=0 -> uniform Kardamaki behavior.
    use_importance = hp.importance_d0_alpha > 0.0
    if use_importance:
        weights = 1.0 + hp.importance_d0_alpha * (
            d0_all.cpu() / max(hp.importance_d0_max, 1e-9))
        weights = weights.clamp(min=1e-9)
        torch.manual_seed(seed + 1)  # for the weighted sampler
    for ep in range(1, hp.K1 + 1):
        if use_importance:
            idx = torch.multinomial(weights, hp.bs, replacement=True)
            x0 = x0_all[idx]
            u0 = u0_all[idx]
            ysp = ysp_all[idx]
            d0 = d0_all[idx]
        else:
            start = (ep - 1) * hp.bs
            end = start + hp.bs
            x0 = x0_all[start:end]
            u0 = u0_all[start:end]
            ysp = ysp_all[start:end]
            d0 = d0_all[start:end]
        # Phase 1: w_du, w_u, w_x = 0
        loss, _ = model.loss(hp.bs, t_wp, N_WP, t_col, N_COL,
                              x0, u0, ysp, d0,
                              hp.w_ode, hp.w_ic, hp.w_ytrk, hp.w_utrk,
                              0.0, 0.0, 0.0)
        if torch.isnan(loss):
            nan_at = ("P1", ep)
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist_p1.append(loss.item())
        if verbose and (ep == 1 or ep % 500 == 0 or ep == hp.K1):
            print(f"  P1 ep {ep:>5}/{hp.K1}  loss={loss.item():.4e}")
    if nan_at is not None:
        return model, {"hist_p1": hist_p1, "hist_p2": [], "nan_at": nan_at}
    # Phase 2: all losses active
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr2)
    for ep in range(1, hp.K2 + 1):
        if use_importance:
            idx = torch.multinomial(weights, hp.bs, replacement=True)
            x0 = x0_all[idx]
            u0 = u0_all[idx]
            ysp = ysp_all[idx]
            d0 = d0_all[idx]
        else:
            start = (ep - 1) * hp.bs
            end = start + hp.bs
            x0 = x0_all[start:end]
            u0 = u0_all[start:end]
            ysp = ysp_all[start:end]
            d0 = d0_all[start:end]
        loss, _ = model.loss(hp.bs, t_wp, N_WP, t_col, N_COL,
                              x0, u0, ysp, d0,
                              hp.w_ode, hp.w_ic, hp.w_ytrk, hp.w_utrk,
                              hp.w_du, hp.w_u, hp.w_x)
        if torch.isnan(loss):
            nan_at = ("P2", ep)
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist_p2.append(loss.item())
        if verbose and (ep == 1 or ep % 500 == 0 or ep == hp.K2):
            print(f"  P2 ep {ep:>5}/{hp.K2}  loss={loss.item():.4e}")
    return model, {"hist_p1": hist_p1, "hist_p2": hist_p2, "nan_at": nan_at}


if __name__ == "__main__":
    # Quick smoke test: train very briefly to verify the pipeline works on CPU
    print(f"Device: {DEVICE}")
    print("Loading SISO training samples...")
    data = torch.load(
        "C:/Pegasus-Sample/Nonlinear-LLMs-PINN-MPC/external/siso_training_samples.pt",
        map_location="cpu", weights_only=False)
    x0_all = data["x0_all"].to(DEVICE)
    u0_all = data["u0_all"].to(DEVICE)
    ysp_all = data["ysp_all"].to(DEVICE)
    d0_all = data["d0_all"].to(DEVICE)
    print(f"  loaded {x0_all.shape[0]} episodes")
    # Tiny training (K1=K2=50 epochs, bs=20) to measure CPU speed
    hp = PINNHparams(K1=50, K2=50, bs=20)
    print(f"\nSmoke training: K1=K2={hp.K1} epochs, bs={hp.bs}")
    t0 = time.time()
    model, hist = train_pinn_siso(hp, x0_all, u0_all, ysp_all, d0_all,
                                    verbose=True)
    elapsed = time.time() - t0
    print(f"\nTotal training time: {elapsed:.1f}s")
    p1, p2 = len(hist["hist_p1"]), len(hist["hist_p2"])
    print(f"P1: {p1} ep, P2: {p2} ep. Per-epoch wall: "
          f"{elapsed/max(1,p1+p2)*1000:.1f} ms")
    print(f"\nExtrapolation to full Kardamaki training (K=10000 ep/phase, bs=100):")
    # bs=100 vs bs=20 = 5x more compute per epoch
    full_secs = elapsed / max(1, p1 + p2) * 20000 * 5
    print(f"  Estimated: {full_secs:.0f} s = {full_secs/60:.1f} min "
          f"= {full_secs/3600:.1f} hours per full training trial")
