"""Phase C: Closed-loop refinement of a trained PINN-MPC.

Two methods, both novel relative to Kardamaki 2026:

  1. Differentiable Closed-Loop refinement (DPC-inspired)
     ----------------------------------------------------
     Kardamaki et al. 2026 train PINN-MPC on PER-EPISODE losses
     (one (x0, sp) pair per training episode). The training signal does NOT
     see the receding-horizon closed-loop behavior.

     We add a refinement stage AFTER their training: roll out the PINN as a
     closed-loop controller over multi-step trajectories, compute the
     accumulated tracking + control-effort cost, and backpropagate
     through the differentiable plant + PINN to fine-tune the weights.

     This matches the Differentiable Predictive Control (DPC) idea
     [Drgona et al. 2022 JPC] applied as a refinement step, not as the
     primary training method.

  2. Residual PPO refinement (kept as fallback baseline)
     ----------------------------------------------------
     A small ResidualNet predicts delta_u; final action u = u_PINN + alpha*delta_u.
     Trained via PPO on closed-loop reward. Safer (residual is bounded)
     and lets the PINN's hard-won physics consistency stay intact.

References:
  [1] Kardamaki et al. (2026). J. Process Control 158, 103634.
  [2] Drgona, J. et al. (2022). "Differentiable Predictive Control."
      J. Process Control 116, 80-92.
  [3] Schulman, J. et al. (2017). "Proximal Policy Optimization." arXiv:1707.06347.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .pinn_siso import DEVICE, PINN_Controller
from .evaluate import EvalScenario, sample_episodes


# ---------------------------------------------------------------------------
# Method 1: Differentiable Closed-Loop refinement (the "DPC" route)
# ---------------------------------------------------------------------------

def differentiable_rollout(model: PINN_Controller,
                             x0: torch.Tensor, ysp: torch.Tensor,
                             u0: torch.Tensor, d0: torch.Tensor,
                             scen: EvalScenario,
                             retain_grad: bool = True) -> dict:
    """Closed-loop rollout that KEEPS the autograd graph through the plant
    + the PINN. Enables direct backprop of trajectory-level loss into PINN
    weights.

    Returns:
      X    (B, T)  state trajectory (differentiable)
      U    (B, T)  control trajectory (differentiable)
      YSP  (B, T)  setpoint at each step
      offset_T (B,) final |y - sp| (differentiable wrt PINN weights)
    """
    bs = x0.shape[0]
    n_steps = int(scen.sim_time / scen.Ts)
    n_inner = int(scen.Ts / scen.dt)

    x_curr = x0.clone()
    u_prev = u0.clone() if u0 is not None else \
        scen.K_VALVE * torch.sqrt(x_curr.clamp(min=0.0))

    X_log, U_log, YSP_log = [], [], []
    t_curr = 0.0
    for _ in range(n_steps):
        ysp_k = x0 if t_curr < scen.sp_change else ysp
        d_k = torch.zeros_like(d0) if t_curr < scen.d_step else d0
        t_query = torch.full((bs,), scen.Ts, device=DEVICE)
        _, u_pred = model(t_query, x_curr, u_prev, ysp_k, d_k)
        u_cmd = u_pred
        # Inner Euler integration (differentiable)
        x_inner = x_curr
        for _ in range(n_inner):
            sqrt_x = torch.sqrt(x_inner.clamp(min=1e-8))   # avoid sqrt(0) NaN
            dx = (u_cmd + d_k - scen.K_VALVE * sqrt_x) / scen.A
            x_inner = (x_inner + dx * scen.dt).clamp(min=0.0)
            t_curr += scen.dt
        X_log.append(x_inner.unsqueeze(1))
        U_log.append(u_cmd.unsqueeze(1))
        YSP_log.append(ysp_k.unsqueeze(1))
        x_curr = x_inner
        u_prev = u_cmd
    X = torch.cat(X_log, dim=1)
    U = torch.cat(U_log, dim=1)
    YSP = torch.cat(YSP_log, dim=1)
    offset_T = (X[:, -1] - ysp).abs()
    return {"X": X, "U": U, "YSP": YSP, "offset_T": offset_T}


@dataclass
class DPCRefineCfg:
    """Hparams for differentiable closed-loop refinement.

    Modes:
      - 'mixed':       2/3 tracking + 1/3 disturbance episodes (original behavior)
      - 'tracking':    tracking-only episodes (d0 = 0)
      - 'disturbance': disturbance-only episodes (d0 sampled from [d_lo, d_hi])

    The two-phase pattern: run 'tracking' first to refine baseline tracking,
    then run 'disturbance' at lower lr to attack the disturbance gap without
    disturbing tracking gains.
    """
    epochs: int = 200              # refinement epochs (vs Kardamaki's 10000)
    bs: int = 32                   # episodes per batch (CL rollouts are expensive)
    lr: float = 1e-5               # very small - refine, don't disrupt
    w_offset: float = 100.0        # final-state tracking
    w_iae: float = 1.0             # integrated abs error over trajectory
    w_smooth: float = 0.01         # input rate-of-change
    w_bounds: float = 10.0         # soft penalty for x/u out of bounds
    seed: int = 0
    early_stop_patience: int = 20  # stop if no improvement
    mode: str = "mixed"            # 'mixed' | 'tracking' | 'disturbance'
    disturbance_oversample: float = 1.0  # weight high-d0 episodes by this factor


def refine_with_dpc(model: PINN_Controller,
                     cfg: DPCRefineCfg | None = None,
                     scen: EvalScenario | None = None,
                     verbose: bool = False,
                     train_data: tuple | None = None,
                     ) -> tuple[PINN_Controller, dict]:
    """Differentiable closed-loop refinement of a pre-trained PINN-MPC.

    The PINN parameters are updated by backprop through the closed-loop
    trajectory. Returns the refined model and a training history.

    Args:
      train_data: Optional (x0_all, u0_all, ysp_all, d0_all) tensors. When
                  provided, DPC samples episodes from these (matching the
                  PINN's training distribution + Kardamaki's evaluation
                  distribution). When omitted, falls back to the legacy
                  `sample_episodes` random sampler (which has a slight
                  distribution shift relative to Kardamaki - typically hurts
                  performance on well-tuned models).
    """
    cfg = cfg or DPCRefineCfg()
    scen = scen or EvalScenario()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Prepare mode-filtered indices once if training data is provided.
    track_idx = dist_idx = mixed_idx = None
    if train_data is not None:
        x0_all, u0_all, ysp_all, d0_all = train_data
        # tracking: d0 ≈ 0
        track_idx = torch.where(d0_all.cpu() < 1e-6)[0]
        dist_idx = torch.where(d0_all.cpu() >= 1e-6)[0]
        mixed_idx = torch.arange(x0_all.shape[0])
        if verbose:
            print(f"  DPC training pool: {len(track_idx)} tracking, "
                  f"{len(dist_idx)} disturbance, {len(mixed_idx)} total")

    hist = {"loss": [], "offset_mean": [], "offset_max": []}
    best_loss = float("inf")
    bad_streak = 0
    for ep in range(1, cfg.epochs + 1):
        # Sample episodes according to mode
        if train_data is not None:
            # Use Kardamaki training distribution - this matches the
            # evaluation distribution and prevents the distribution shift
            # that wrecks well-tuned models.
            if cfg.mode == "tracking":
                pool = track_idx
            elif cfg.mode == "disturbance":
                pool = dist_idx
            else:
                pool = mixed_idx
            sel = pool[torch.randint(0, len(pool), (cfg.bs,))]
            x0 = x0_all[sel]
            u0 = u0_all[sel]
            ysp = ysp_all[sel]
            d0 = d0_all[sel]
            if cfg.mode == "disturbance" and cfg.disturbance_oversample > 1.0:
                k = int(cfg.bs * (cfg.disturbance_oversample - 1.0))
                k = min(k, cfg.bs)
                top_idx = torch.topk(d0, k).indices
                x0 = torch.cat([x0, x0[top_idx]])
                ysp = torch.cat([ysp, ysp[top_idx]])
                u0 = torch.cat([u0, u0[top_idx]])
                d0 = torch.cat([d0, d0[top_idx]])
        else:
            # Legacy path - sample_episodes (slight distribution shift)
            if cfg.mode == "tracking":
                eps = sample_episodes(cfg.bs, scen, mode="tracking", seed=ep)
                x0, ysp, u0, d0 = eps["x0"], eps["ysp"], eps["u0"], eps["d0"]
            elif cfg.mode == "disturbance":
                eps = sample_episodes(cfg.bs, scen, mode="disturbance",
                                        seed=ep + 10000)
                x0, ysp, u0, d0 = eps["x0"], eps["ysp"], eps["u0"], eps["d0"]
                if cfg.disturbance_oversample > 1.0:
                    k = int(cfg.bs * (cfg.disturbance_oversample - 1.0))
                    k = min(k, cfg.bs)
                    top_idx = torch.topk(d0, k).indices
                    x0 = torch.cat([x0, x0[top_idx]])
                    ysp = torch.cat([ysp, ysp[top_idx]])
                    u0 = torch.cat([u0, u0[top_idx]])
                    d0 = torch.cat([d0, d0[top_idx]])
            else:
                eps = sample_episodes(cfg.bs, scen, mode="tracking", seed=ep)
                eps_d = sample_episodes(cfg.bs // 2, scen, mode="disturbance",
                                          seed=ep + 10000)
                x0 = torch.cat([eps["x0"], eps_d["x0"]])
                ysp = torch.cat([eps["ysp"], eps_d["ysp"]])
                u0 = torch.cat([eps["u0"], eps_d["u0"]])
                d0 = torch.cat([eps["d0"], eps_d["d0"]])

        # Rollout WITH gradients
        roll = differentiable_rollout(model, x0, ysp, u0, d0, scen)
        X, U, YSP = roll["X"], roll["U"], roll["YSP"]

        # Composite trajectory loss
        offset_final = (X[:, -1] - ysp).pow(2).mean()
        iae = (X - YSP).abs().mean()
        smooth = (U[:, 1:] - U[:, :-1]).pow(2).mean()
        bound_violation = (
            torch.relu(X - 4.0).pow(2).mean()     # x_max=4
            + torch.relu(0.0 - X).pow(2).mean()
            + torch.relu(U - 1.0).pow(2).mean()   # u_max=1
            + torch.relu(0.0 - U).pow(2).mean()
        )
        loss = (cfg.w_offset * offset_final
                 + cfg.w_iae * iae
                 + cfg.w_smooth * smooth
                 + cfg.w_bounds * bound_violation)

        opt.zero_grad()
        loss.backward()
        # Clip gradients to keep refinement gentle
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        offset_mean = float((X[:, -1] - ysp).abs().mean().item())
        offset_max = float((X[:, -1] - ysp).abs().max().item())
        hist["loss"].append(float(loss.item()))
        hist["offset_mean"].append(offset_mean)
        hist["offset_max"].append(offset_max)
        if verbose and (ep == 1 or ep % 20 == 0):
            print(f"  DPC refine ep {ep:>4}/{cfg.epochs}  "
                  f"loss={loss.item():.4e}  "
                  f"offset_mean={offset_mean:.4f} m  "
                  f"offset_max={offset_max:.4f} m")
        if loss.item() < best_loss - 1e-5:
            best_loss = loss.item()
            bad_streak = 0
        else:
            bad_streak += 1
            if bad_streak >= cfg.early_stop_patience:
                if verbose:
                    print(f"  Early stop at ep {ep} "
                          f"(no improvement for {bad_streak} eps)")
                break

    model.eval()
    return model, hist


# ---------------------------------------------------------------------------
# Method 2: PPO with residual policy (kept as alternate path)
# ---------------------------------------------------------------------------

class ResidualPolicy(nn.Module):
    """Small MLP that outputs a small additive correction to the PINN's u.

    Final action = u_PINN + alpha * tanh(residual(state)) * delta_max

    The bounded residual (via tanh + scaling) keeps refinement safe.
    """
    def __init__(self, hidden: int = 32, delta_max: float = 0.1):
        super().__init__()
        self.delta_max = delta_max
        # Input: (x, u_PINN, ysp, d) -> small delta_u
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, u_pinn: torch.Tensor,
                ysp: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        inp = torch.stack((x, u_pinn, ysp, d), dim=-1)
        return self.delta_max * torch.tanh(self.net(inp).squeeze(-1))


@dataclass
class ResidualCfg:
    hidden: int = 32
    delta_max: float = 0.1
    epochs: int = 200
    bs: int = 32
    lr: float = 5e-4
    seed: int = 0


def refine_with_residual(
    pinn: PINN_Controller,
    cfg: ResidualCfg | None = None,
    scen: EvalScenario | None = None,
    verbose: bool = False,
) -> tuple[ResidualPolicy, dict]:
    """Train a residual policy on top of a frozen PINN-MPC.

    This is conceptually similar to RL but uses direct backprop through
    the differentiable plant (no PPO needed). PINN stays frozen.
    """
    cfg = cfg or ResidualCfg()
    scen = scen or EvalScenario()
    torch.manual_seed(cfg.seed)

    # Freeze PINN
    for p in pinn.parameters():
        p.requires_grad_(False)
    pinn.eval()

    residual = ResidualPolicy(hidden=cfg.hidden,
                                delta_max=cfg.delta_max).to(DEVICE)
    opt = torch.optim.Adam(residual.parameters(), lr=cfg.lr)

    hist = {"loss": [], "offset_mean": []}
    for ep in range(1, cfg.epochs + 1):
        eps = sample_episodes(cfg.bs, scen, mode="tracking", seed=ep)
        x0 = eps["x0"]; ysp = eps["ysp"]; u0 = eps["u0"]; d0 = eps["d0"]
        bs = x0.shape[0]
        n_steps = int(scen.sim_time / scen.Ts)
        n_inner = int(scen.Ts / scen.dt)
        x_curr = x0.clone()
        u_prev = u0.clone()
        traj_offset = 0.0
        t_curr = 0.0
        for _ in range(n_steps):
            ysp_k = x0 if t_curr < scen.sp_change else ysp
            d_k = torch.zeros_like(d0) if t_curr < scen.d_step else d0
            with torch.no_grad():
                t_query = torch.full((bs,), scen.Ts, device=DEVICE)
                _, u_pinn = pinn(t_query, x_curr, u_prev, ysp_k, d_k)
            delta_u = residual(x_curr, u_pinn, ysp_k, d_k)
            u_cmd = (u_pinn + delta_u).clamp(0.0, 1.0)
            x_inner = x_curr
            for _ in range(n_inner):
                sqrt_x = torch.sqrt(x_inner.clamp(min=1e-8))
                dx = (u_cmd + d_k - scen.K_VALVE * sqrt_x) / scen.A
                x_inner = (x_inner + dx * scen.dt).clamp(min=0.0)
                t_curr += scen.dt
            x_curr = x_inner
            u_prev = u_cmd
        final_offset = (x_curr - ysp).pow(2).mean()
        opt.zero_grad()
        final_offset.backward()
        opt.step()
        hist["loss"].append(float(final_offset.item()))
        offset_mean = float((x_curr - ysp).abs().mean().item())
        hist["offset_mean"].append(offset_mean)
        if verbose and (ep == 1 or ep % 20 == 0):
            print(f"  Residual ep {ep:>4}/{cfg.epochs}  "
                  f"loss={final_offset.item():.4e}  "
                  f"offset_mean={offset_mean:.4f} m")
    return residual, hist


if __name__ == "__main__":
    # Smoke test: train a tiny PINN, then refine it with DPC, see if offset drops
    from .pinn_siso import PINNHparams, train_pinn_siso
    from .evaluate import evaluate_model
    print(f"Device: {DEVICE}")
    print("Loading SISO training samples...")
    data = torch.load(
        "C:/Pegasus-Sample/Nonlinear-LLMs-PINN-MPC/external/siso_training_samples.pt",
        map_location="cpu", weights_only=False)
    x0_all = data["x0_all"].to(DEVICE)
    u0_all = data["u0_all"].to(DEVICE)
    ysp_all = data["ysp_all"].to(DEVICE)
    d0_all = data["d0_all"].to(DEVICE)
    # Pre-train PINN at reduced scale
    hp = PINNHparams(K1=300, K2=300, bs=50)
    print(f"\n1. Pre-training PINN-MPC ({hp.K1}+{hp.K2} epochs, bs={hp.bs})...")
    t0 = time.time()
    model, _ = train_pinn_siso(hp, x0_all, u0_all, ysp_all, d0_all,
                                 verbose=False)
    print(f"   Pre-train time: {time.time()-t0:.1f}s")
    print("\n2. BEFORE refinement:")
    m_before = evaluate_model(model, n_tracking=200, n_disturbance=200)
    for k in ["tracking_mean_offset_m", "tracking_max_offset_m",
                "disturbance_mean_offset_m", "disturbance_max_offset_m"]:
        print(f"   {k}: {m_before[k]:.4f}")
    print("\n3. DPC refinement (100 epochs)...")
    cfg = DPCRefineCfg(epochs=100, bs=32, lr=1e-5)
    t0 = time.time()
    model, hist = refine_with_dpc(model, cfg, verbose=True)
    print(f"   Refine time: {time.time()-t0:.1f}s")
    print("\n4. AFTER refinement:")
    m_after = evaluate_model(model, n_tracking=200, n_disturbance=200)
    for k in ["tracking_mean_offset_m", "tracking_max_offset_m",
                "disturbance_mean_offset_m", "disturbance_max_offset_m"]:
        before = m_before[k]
        after = m_after[k]
        delta = (after - before) / max(1e-6, before) * 100
        print(f"   {k}: {before:.4f} -> {after:.4f}  "
              f"({delta:+.1f}%)")
