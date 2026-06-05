"""Residual-Physics NARX (delta-PINN) for the Thosar 2025 CSTR.

NARX-shaped wrapper around an explicit physics step:

  y_hat(t) = physics_step(y(t-1), u(t-1))  +  NN(y(t-w), ..., y(t-1), u(t-1))

The NN now learns only the model-plant mismatch (residual / discrepancy)
instead of the full forward map. For systems where the physics model is
approximately right, this:
  - Bounds the function the NN has to learn (small smooth residual),
  - Inherits good extrapolation from the physics step,
  - Acts as a hard inductive bias instead of a soft penalty (paper PI-NARX).

References:
  - Bhouri & Perdikaris (2023), "Gaussian processes meet NeuralODEs"
  - Wang et al. (2024), "Discrepancy networks for hybrid modeling"
  - Bikmukhametov & Jaschke (2020), "Combining machine learning and process
    engineering physics" (parallel hybrid models)

The physics step uses RK4 with `sub_steps` substeps to stay accurate on the
stiff T equation (Arrhenius). All evaluations are done in ORIGINAL (un-scaled)
units; the residual is computed in NORMALIZED [-1, 1] space so the NN sees
inputs and outputs on the same scale as the NARX baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from nn_utils import (MLP, MinMaxNorm, make_windows)
from pinarx_plant import CSTRParams, step as scipy_step


# ============================================================================
# Torch-native plant rhs (Eqs 6-9). Differentiable but used only as a
# numpy-equivalent here since the physics target is computed via scipy LSODA.
# Kept available for callers that need rhs in autograd context.
# ============================================================================
def torch_rhs(y: torch.Tensor, u: torch.Tensor,
              p: CSTRParams) -> torch.Tensor:
    C_A, T, T_c, h = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    Q_f, Q_c       = u[:, 0], u[:, 1]
    h_safe = torch.clamp(h, min=p.h_min)
    sqrt_h = torch.sqrt(h_safe)
    Vol    = p.A * h_safe
    k  = p.k_0 * torch.exp(-p.Ea_over_R / torch.clamp(T, min=1.0))
    rA = k * C_A
    dC_A = -rA + (Q_f * p.C_Af - p.C_V * sqrt_h * C_A) / Vol
    Qrxn = rA * (-p.delta_H) / p.rho_Cp
    Qconv = (Q_f * p.T_f - p.C_V * sqrt_h * T) / Vol
    Qjacket = p.UA_c * (T_c - T) / (p.rho_Cp * Vol)
    dT    = Qrxn + Qconv + Qjacket
    dT_c  = (Q_c * (p.T_cf - T_c) / p.V_c
              + p.UA_c * (T - T_c) / (p.rho_c_Cpc * p.V_c))
    dh    = (Q_f - p.C_V * sqrt_h) / p.A
    return torch.stack([dC_A, dT, dT_c, dh], dim=1)


# ============================================================================
# Hyperparameters - all knobs the LEAN tuner / LLM optimizer can change
# ============================================================================
@dataclass
class ResPhysNARXHparams:
    # --- Architecture ---
    window:     int    = 2
    hidden:     tuple  = (200, 400, 200)
    activation: str    = "tanh"     # tanh | relu | gelu | silu
    include_u_lags: bool = False    # False = u(t-1) only (10-D input for w=2)

    # --- Optimizer ---
    lr_adam:           float = 1e-3
    lr_lbfgs:          float = 0.1
    n_epochs_adam:     int   = 1000
    batch_size:        int   = 64
    lbfgs_iters:       int   = 1000
    weight_decay:      float = 0.0
    early_stop_patience: int = 100

    # --- Res-Phys specific ---
    # NOTE: physics target is computed offline once per training run via
    # scipy LSODA (see _phys_step_normalized), not via torch RK4. The
    # rk4_sub_steps knob is kept for the case where a user wants to swap
    # the offline solver back to torch RK4 inside autograd.
    rk4_sub_steps: int  = 50
    # Optional regularizer that encourages the NN residual to stay small
    # ("trust physics"). 0.0 disables.
    residual_l2:   float = 0.0
    # Limited-knowledge ablation (paper Table 5 equivalent).
    #   full   = all 4 channels predicted by physics (mass+energy+level)
    #   mass   = physics knows C_A, h (mass+level); T, T_c left to NN
    #   energy = physics knows T, T_c (energy);     C_A, h left to NN
    #   none   = pure NARX (physics = identity); NN learns everything
    physics_mode: str = "full"

    # --- Bookkeeping ---
    seed:   int = 0
    device: str = "cpu"


# ============================================================================
# Torch-native RK4 step over the plant ODE (no grad: physics is target-only)
# ============================================================================
def torch_rk4_step(y: torch.Tensor, u: torch.Tensor,
                    dt: float, p: CSTRParams, sub_steps: int = 10
                    ) -> torch.Tensor:
    """Vectorized RK4 over dt minutes with `sub_steps` substeps.

    Input shapes: y (B, 4), u (B, 2). Returns y(t+dt) (B, 4).
    Constant `u` (zero-order hold) over the interval.
    Runs with grad disabled by callers (physics is a fixed target).
    """
    h = dt / sub_steps
    for _ in range(sub_steps):
        k1 = torch_rhs(y,            u, p)
        k2 = torch_rhs(y + 0.5*h*k1, u, p)
        k3 = torch_rhs(y + 0.5*h*k2, u, p)
        k4 = torch_rhs(y +     h*k3, u, p)
        y  = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Physical floors (mirror pinarx_plant.step)
        y = torch.stack([
            torch.clamp(y[:, 0], min=0.0),
            torch.clamp(y[:, 1], min=1.0),
            torch.clamp(y[:, 2], min=1.0),
            torch.clamp(y[:, 3], min=p.h_min),
        ], dim=1)
    return y


# ============================================================================
# Residual-Physics NARX model
# ============================================================================
class ResPhysNARXModel:
    """NARX MLP wrapped around an RK4 physics step.

    The NN learns the *residual* (in [-1, 1] normalized space) between the
    physics-predicted next state and the true next state.
    """
    def __init__(self, n_u: int, n_y: int, hp: ResPhysNARXHparams,
                  p_plant: CSTRParams | None = None):
        self.n_u, self.n_y, self.hp = n_u, n_y, hp
        self.p_plant = p_plant or CSTRParams()
        u_dim = hp.window * n_u if hp.include_u_lags else n_u
        self.in_dim  = hp.window * n_y + u_dim
        self.out_dim = n_y
        self.x_norm = MinMaxNorm()
        self.y_norm = MinMaxNorm()
        torch.manual_seed(hp.seed)
        np.random.seed(hp.seed)
        self.net = MLP(self.in_dim, self.out_dim,
                        hidden=hp.hidden, activation=hp.activation
                        ).to(hp.device)

    # ------------------------------------------------------------------
    # Helper: physics step in normalized [-1, 1] target space
    #
    # Uses scipy LSODA (via pinarx_plant.step) per-sample. We don't need
    # autodiff through the physics target (the NN only sees it as a fixed
    # offset), and scipy LSODA handles the stiff Arrhenius reliably for
    # all training amplitudes. ~1ms/sample, called once before training.
    # ------------------------------------------------------------------
    def _phys_step_normalized(self, y_prev_orig: torch.Tensor,
                                 u_orig: torch.Tensor) -> torch.Tensor:
        """y_prev_orig: (B, n_y) ORIGINAL units. u_orig: (B, n_u) ORIGINAL units.
        Returns y_phys NORMALIZED to [-1, 1] using self.y_norm.
        """
        y_prev_np = y_prev_orig.detach().cpu().numpy().astype(np.float64)
        u_np      = u_orig.detach().cpu().numpy().astype(np.float64)
        B = y_prev_np.shape[0]

        # Sanitize before scipy. Autoregressive rollout on a poorly trained
        # model can drift to NaN/Inf, which makes scipy_step raise
        # "All components of y0 must be finite". Replace non-finite entries
        # with a physically-safe state and clamp to sensible bounds so the
        # rollout keeps going (the resulting MAE will be large -> still a
        # useful signal that the config is unstable).
        if not np.isfinite(y_prev_np).all():
            safe = np.array([0.001, 350.0, 320.0, 1.0], dtype=np.float64)
            mask = ~np.isfinite(y_prev_np)
            y_prev_np = np.where(mask, np.broadcast_to(safe, y_prev_np.shape),
                                  y_prev_np)
        y_prev_np[:, 0] = np.clip(y_prev_np[:, 0], 0.0,    10.0)
        y_prev_np[:, 1] = np.clip(y_prev_np[:, 1], 200.0, 600.0)
        y_prev_np[:, 2] = np.clip(y_prev_np[:, 2], 200.0, 600.0)
        y_prev_np[:, 3] = np.clip(y_prev_np[:, 3], 0.01,   50.0)
        if not np.isfinite(u_np).all():
            u_safe = np.array([120.0, 15.0], dtype=np.float64)
            u_np = np.where(np.isfinite(u_np),
                             u_np, np.broadcast_to(u_safe, u_np.shape))

        mode = getattr(self.hp, "physics_mode", "full")
        if mode == "none":
            # NN learns everything; physics is identity. Equivalent to plain NARX.
            y_phys_np = y_prev_np.astype(np.float32)
        else:
            y_phys_np = np.empty((B, self.n_y), dtype=np.float32)
            for i in range(B):
                y_phys_np[i] = scipy_step(y_prev_np[i], u_np[i],
                                             dt=1.0, p=self.p_plant)
            if mode == "mass":
                # Knock out energy channels (T, T_c) -> physics doesn't predict them.
                y_phys_np[:, 1] = y_prev_np[:, 1].astype(np.float32)
                y_phys_np[:, 2] = y_prev_np[:, 2].astype(np.float32)
            elif mode == "energy":
                # Knock out mass/level channels (C_A, h) -> physics doesn't predict them.
                y_phys_np[:, 0] = y_prev_np[:, 0].astype(np.float32)
                y_phys_np[:, 3] = y_prev_np[:, 3].astype(np.float32)
            elif mode != "full":
                raise ValueError(f"Unknown physics_mode={mode!r}; "
                                  f"choose full | mass | energy | none")
        y_phys = torch.from_numpy(y_phys_np).to(y_prev_orig.device)
        y_mean = torch.from_numpy(self.y_norm.mean).to(y_phys.device)
        y_std  = torch.from_numpy(self.y_norm.std ).to(y_phys.device)
        return (y_phys - y_mean) / y_std

    # ------------------------------------------------------------------
    # Forward: NN residual + physics step  (NORMALIZED output space)
    # ------------------------------------------------------------------
    def _forward_norm(self, x_norm: torch.Tensor,
                        y_prev_orig: torch.Tensor,
                        u_orig: torch.Tensor) -> torch.Tensor:
        """Returns y_hat in NORMALIZED [-1, 1] space, with grad through the NN
        (residual) but NOT through the physics step (treated as target).
        """
        residual_n = self.net(x_norm)                      # (B, n_y) in [-1,1]ish
        y_phys_n   = self._phys_step_normalized(y_prev_orig, u_orig)
        return y_phys_n + residual_n

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, traj_train: dict, traj_val: dict | None = None,
            verbose: bool = True) -> dict:
        hp = self.hp
        device = hp.device

        # 1. Windowed datasets
        X_tr, Y_tr = make_windows(traj_train["u"], traj_train["y"], hp.window,
                                    include_u_lags=hp.include_u_lags)
        if traj_val is not None:
            X_va, Y_va = make_windows(traj_val["u"], traj_val["y"], hp.window,
                                        include_u_lags=hp.include_u_lags)

        # 2. Normalize. Same recipe as NARX: theoretical u bounds + train+val
        # empirical y bounds.
        if traj_val is not None:
            X_fit = np.concatenate([X_tr, X_va], axis=0)
            Y_fit = np.concatenate([Y_tr, Y_va], axis=0)
        else:
            X_fit, Y_fit = X_tr, Y_tr
        self.x_norm.fit(X_fit); self.y_norm.fit(Y_fit)
        u_start = hp.window * self.n_y
        u_lo_theory = np.array([100.0, 10.0], dtype=np.float32)
        u_hi_theory = np.array([140.0, 20.0], dtype=np.float32)
        if hp.include_u_lags:
            u_lo_theory = np.tile(u_lo_theory, hp.window)
            u_hi_theory = np.tile(u_hi_theory, hp.window)
        self.x_norm.lo[u_start:] = u_lo_theory
        self.x_norm.hi[u_start:] = u_hi_theory
        self.x_norm.mean[u_start:] = (u_lo_theory + u_hi_theory) / 2.0
        self.x_norm.std[u_start:]  = (u_hi_theory - u_lo_theory) / 2.0

        # 3. Tensors. We need (a) normalized X for the NN and (b) the
        # PHYSICS TARGET y_phys for every sample. y_phys depends only on
        # (y_prev, u_prev) which are fixed inputs, NOT on NN weights - so
        # we pre-compute them once here and cache. Avoids running RK4
        # inside every training batch (would be 1000s of times slower).
        X_tr_n = self.x_norm.transform(X_tr)
        Y_tr_n = self.y_norm.transform(Y_tr)
        X_tr_t = torch.from_numpy(X_tr_n).to(device)
        Y_tr_t = torch.from_numpy(Y_tr_n).to(device)

        y_prev_tr = torch.from_numpy(X_tr[:, :self.n_y].astype(np.float32)
                                       ).to(device)
        u_tr = torch.from_numpy(X_tr[:, u_start:u_start + self.n_u
                                       ].astype(np.float32)).to(device)

        if verbose:
            print(f"  pre-computing physics target on {len(X_tr_t)} train "
                    f"samples (rk4_substeps={hp.rk4_sub_steps})...")
        with torch.no_grad():
            y_phys_tr_n = self._phys_step_normalized(y_prev_tr, u_tr).detach()

        if traj_val is not None:
            X_va_t = torch.from_numpy(self.x_norm.transform(X_va)).to(device)
            Y_va_t = torch.from_numpy(self.y_norm.transform(Y_va)).to(device)
            y_prev_va = torch.from_numpy(X_va[:, :self.n_y].astype(np.float32)
                                            ).to(device)
            u_va = torch.from_numpy(X_va[:, u_start:u_start + self.n_u
                                           ].astype(np.float32)).to(device)
            with torch.no_grad():
                y_phys_va_n = self._phys_step_normalized(y_prev_va, u_va
                                                             ).detach()

        # 4. Adam phase with early stopping
        opt = torch.optim.Adam(self.net.parameters(),
                                lr=hp.lr_adam, weight_decay=hp.weight_decay)
        n = X_tr_t.shape[0]
        hist = {"adam_loss": [], "val_loss": []}
        best_val   = float("inf")
        best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        epochs_no_improve = 0
        use_es = (hp.early_stop_patience > 0) and (traj_val is not None)

        for ep in range(hp.n_epochs_adam):
            perm = torch.randperm(n, device=device)
            ep_loss = 0.0; n_batches = 0
            for s in range(0, n, hp.batch_size):
                idx = perm[s:s + hp.batch_size]
                # Use cached physics target instead of recomputing RK4
                residual_n = self.net(X_tr_t[idx])
                pred = y_phys_tr_n[idx] + residual_n
                loss = ((pred - Y_tr_t[idx]) ** 2).mean()
                if hp.residual_l2 > 0:
                    res_only = self.net(X_tr_t[idx])
                    loss = loss + hp.residual_l2 * (res_only ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += float(loss.item()); n_batches += 1
            ep_loss /= max(n_batches, 1)
            hist["adam_loss"].append(ep_loss)

            if traj_val is not None:
                with torch.no_grad():
                    vp = y_phys_va_n + self.net(X_va_t)
                    vl = float(((vp - Y_va_t) ** 2).mean().item())
                hist["val_loss"].append(vl)
                if use_es:
                    if vl < best_val - 1e-9:
                        best_val = vl
                        best_state = {k: v.clone() for k, v
                                        in self.net.state_dict().items()}
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            if verbose and (ep % max(1, hp.n_epochs_adam // 10) == 0
                             or ep == hp.n_epochs_adam - 1):
                msg = f"  adam ep {ep:>4}/{hp.n_epochs_adam} train={ep_loss:.4g}"
                if traj_val is not None: msg += f"  val={hist['val_loss'][-1]:.4g}"
                print(msg)

            if use_es and epochs_no_improve >= hp.early_stop_patience:
                if verbose:
                    print(f"  early-stopped at ep {ep+1}")
                break

        if use_es:
            self.net.load_state_dict(best_state)
            hist["best_val_adam"] = best_val

        # 5. L-BFGS (data only - physics is fixed target)
        if hp.lbfgs_iters > 0:
            lbfgs = torch.optim.LBFGS(self.net.parameters(),
                                       lr=hp.lr_lbfgs,
                                       max_iter=hp.lbfgs_iters,
                                       history_size=50,
                                       tolerance_grad=1e-9,
                                       tolerance_change=1e-12,
                                       line_search_fn="strong_wolfe")
            def closure():
                lbfgs.zero_grad()
                residual_n = self.net(X_tr_t)
                pred = y_phys_tr_n + residual_n
                L = ((pred - Y_tr_t) ** 2).mean()
                if hp.residual_l2 > 0:
                    L = L + hp.residual_l2 * (residual_n ** 2).mean()
                L.backward()
                return L
            final = float(lbfgs.step(closure).item())
            hist["lbfgs_final_loss"] = final
            if verbose:
                print(f"  lbfgs done    loss={final:.6g}")
                if traj_val is not None:
                    with torch.no_grad():
                        vp = y_phys_va_n + self.net(X_va_t)
                        vl = float(((vp - Y_va_t) ** 2).mean().item())
                    print(f"  lbfgs final   val={vl:.6g}")
                    hist["val_loss_final"] = vl

        return hist

    # ------------------------------------------------------------------
    # Autoregressive rollout (uses model's own predictions as history)
    # ------------------------------------------------------------------
    def rollout(self, u: np.ndarray, y_init: np.ndarray) -> np.ndarray:
        hp = self.hp
        w  = hp.window
        N  = u.shape[0]
        device = hp.device
        y_pred = np.zeros((N + 1, self.n_y), dtype=np.float32)
        y_pred[:w] = y_init

        self.net.eval()
        with torch.no_grad():
            for t in range(w, N + 1):
                ywin = [y_pred[t - i] for i in range(1, w + 1)]
                if hp.include_u_lags:
                    uwin = [u[t - i] for i in range(1, w + 1)]
                else:
                    uwin = [u[t - 1]]
                feat = np.concatenate(ywin + uwin)[None, :]
                feat_n = self.x_norm.transform(feat)
                feat_t = torch.from_numpy(feat_n).to(device)
                y_prev_t = torch.from_numpy(
                    y_pred[t - 1][None, :].astype(np.float32)).to(device)
                u_t = torch.from_numpy(
                    u[t - 1][None, :].astype(np.float32)).to(device)
                pred_n = self._forward_norm(feat_t, y_prev_t, u_t
                                              ).cpu().numpy()
                pred = self.y_norm.inverse(pred_n)[0]
                # Hard physical clamps - prevents runaway autoregressive
                # divergence from poisoning the windowed input on the next
                # step. Values outside these bounds are non-physical anyway.
                if not np.isfinite(pred).all():
                    pred = np.array([0.001, 350.0, 320.0, 1.0],
                                      dtype=np.float32)
                pred[0] = np.clip(pred[0], 0.0,    10.0)
                pred[1] = np.clip(pred[1], 200.0, 600.0)
                pred[2] = np.clip(pred[2], 200.0, 600.0)
                pred[3] = np.clip(pred[3], 0.01,   50.0)
                y_pred[t] = pred
        self.net.train()
        return y_pred

    def eval_mae(self, traj: dict, return_pred: bool = False
                  ) -> tuple[float, np.ndarray] | float:
        u, y_true = traj["u"], traj["y"]
        w = self.hp.window
        y_pred = self.rollout(u, y_init=y_true[:w])
        y_pred_n = self.y_norm.transform(y_pred[w:])
        y_true_n = self.y_norm.transform(y_true[w:])
        mae = float(np.mean(np.abs(y_pred_n - y_true_n)))
        if return_pred:
            return mae, y_pred
        return mae

    def eval_mae_one_step(self, traj: dict) -> float:
        u, y_true = traj["u"], traj["y"]
        X, Y = make_windows(u, y_true, self.hp.window,
                              include_u_lags=self.hp.include_u_lags)
        device = self.hp.device
        u_start = self.hp.window * self.n_y
        y_prev = torch.from_numpy(X[:, :self.n_y].astype(np.float32)).to(device)
        u_t    = torch.from_numpy(X[:, u_start:u_start + self.n_u
                                     ].astype(np.float32)).to(device)
        X_n = self.x_norm.transform(X)
        X_t = torch.from_numpy(X_n).to(device)
        Y_n_true = self.y_norm.transform(Y)
        self.net.eval()
        with torch.no_grad():
            Y_n_pred = self._forward_norm(X_t, y_prev, u_t).cpu().numpy()
        self.net.train()
        return float(np.mean(np.abs(Y_n_pred - Y_n_true)))


# ============================================================================
# Self-test
# ============================================================================
if __name__ == "__main__":
    import argparse, torch as _t
    from data_gen import (gen_grid_train_val_split, gen_train_val_split,
                            gen_test1_set, gen_test2_set)

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if _t.cuda.is_available() else "cpu")
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--protocol", choices=["grid", "paper-aprbs"], default="grid")
    ap.add_argument("--qf-levels", type=int, default=10)
    ap.add_argument("--qc-levels", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lbfgs",  type=int, default=1000)
    ap.add_argument("--rk4-sub-steps", type=int, default=50)
    ap.add_argument("--residual-l2",   type=float, default=0.0)
    args = ap.parse_args()

    print("=" * 70)
    print("Residual-Physics NARX  (paper Table 2 ref: PI-NARX T1=0.001242, T2=0.01556)")
    print(f"device={args.device}, seed={args.seed}, protocol={args.protocol}, "
          f"rk4_substeps={args.rk4_sub_steps}")
    print("=" * 70)

    if args.protocol == "grid":
        tr, va = gen_grid_train_val_split(qf_levels=args.qf_levels,
                                              qc_levels=args.qc_levels,
                                              seed=args.seed)
    else:
        tr, va = gen_train_val_split(N_total=5000, N_train=2000,
                                        seed=args.seed)
    t1 = gen_test1_set()
    t2 = gen_test2_set()
    print(f"  train={tr['u'].shape[0]}  val={va['u'].shape[0]}")

    hp = ResPhysNARXHparams(
        window=2, hidden=(200, 400, 200), activation="tanh",
        lr_adam=1e-3, n_epochs_adam=args.epochs, batch_size=64,
        lbfgs_iters=args.lbfgs, lr_lbfgs=0.1,
        weight_decay=0.0,
        early_stop_patience=120, seed=args.seed,
        include_u_lags=False, device=args.device,
        rk4_sub_steps=args.rk4_sub_steps,
        residual_l2=args.residual_l2,
    )
    model = ResPhysNARXModel(n_u=2, n_y=4, hp=hp)
    n_params = sum(p.numel() for p in model.net.parameters())
    print(f"Res-Phys NARX: in_dim={model.in_dim}  out_dim={model.out_dim}  "
            f"params={n_params:,}  rk4_substeps={hp.rk4_sub_steps}")
    hist = model.fit(tr, va, verbose=True)

    one1 = model.eval_mae_one_step(t1)
    one2 = model.eval_mae_one_step(t2)
    ar1  = model.eval_mae(t1)
    ar2  = model.eval_mae(t2)
    print()
    print(f"  Test 1 ONE-STEP   MAE = {one1:.6f}  (paper PI-NARX = 0.001242)")
    print(f"  Test 2 ONE-STEP   MAE = {one2:.6f}  (paper PI-NARX = 0.01556)")
    print(f"  Test 1 AUTOREG    MAE = {ar1:.6f}")
    print(f"  Test 2 AUTOREG    MAE = {ar2:.6f}")
