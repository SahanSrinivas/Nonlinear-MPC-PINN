"""PI-NARX (Physics-Informed NARX) - paper Eqs 4-5 + §3.2.

Same network as the baseline NARX, but trained with the composite loss

    L = lambda_l * L_data  +  lambda_p * L_physics                       (Eq 4)

  L_data    = MSE on the (X, Y) windowed training pairs                  (Eq 5)
  L_physics = MSE of the *ODE residual* at random LHS collocation points

The ODE residual uses Eqs 6-9 of the paper (the same Bequette CSTR we use
in pinarx_plant.py) and a forward-Euler discretization at dt = 1 min:

    res_i = ( y_NARX(t)  -  y_coll(t-1) ) / dt  -  rhs( y_coll(t-1),
                                                          u_coll(t-1) )

where (y_coll(t-1), u_coll(t-1)) are LHS samples within the training envelope
and y_NARX(t) is the network's one-step prediction given a w-step window of
identical y_coll(t-1)'s (this approximates a "steady history" collocation point
- a standard trick for NARX physics-informed training).

The loss is computed in *original* (denormalized) state units, so the
residual is dimensionally meaningful; each channel is then scaled by 1/std
to keep all four states comparable.

This module reuses the same MLP, ZScore, and training plumbing as `narx.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from narx import (NARXHparams, MLP, ZScore, make_windows)
from pinarx_plant import CSTRParams
from data_gen import (TRAIN_QF_RANGE, TRAIN_QC_RANGE)


# ============================================================================
# PI-NARX hyperparameters (paper Table 7 + LLM-AutoOpt knobs)
# ============================================================================
@dataclass
class PINARXHparams(NARXHparams):
    # Loss weights from paper §4.1: lambda_l = 10^10 multiplying the
    # data MSE (in normalized units), lambda_p = 0.01 multiplying the
    # physics MSE (computed in ORIGINAL units, per paper's note that
    # "rescale sampled points and compute residuals in the original,
    # non-scaled values").
    lambda_data: float = 1e10
    lambda_phys: float = 1e-2

    # Collocation pts: paper says 10,000 LHS in the 10-D NARX input space
    n_collocation: int = 10_000

    # Forward-Euler dt for the physics residual (paper plant has dt=1 min)
    dt_phys: float = 1.0

    # Which physics terms are enabled (for the limited-knowledge ablation,
    # paper Table 5). When False the corresponding output channel's residual
    # is zeroed: mass=C_A,h; energy=T,T_c.
    phys_mass:   bool = True
    phys_energy: bool = True

    # Paper uses 9000 L-BFGS iters for PI-NARX (vs 1000 for plain NARX)
    lbfgs_iters: int = 9000


# ============================================================================
# Latin Hypercube Sampler in the 10-D NARX input space  (paper §4.1)
#
# Paper: "10,000 data points are generated across 10 dimensions corresponding
# to the inputs". The 10 dims for w=2 are:
#    y(t-1) : 4 channels (C_A, T, T_c, h)
#    y(t-2) : 4 channels (C_A, T, T_c, h)
#    u(t-1) : 2 channels (Q_f, Q_c)
# Per-dim ranges are taken from the empirical min/max of the training
# trajectory (so collocation stays on physically achievable inputs).
# ============================================================================
def _lhs_1d(lo: float, hi: float, n: int, rng) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, n + 1)
    u     = rng.uniform(edges[:-1], edges[1:])
    rng.shuffle(u)
    return lo + (hi - lo) * u


def sample_collocation_narx(traj_train: dict, n: int, w: int = 2,
                              seed: int = 0
                              ) -> tuple[np.ndarray, np.ndarray]:
    """LHS sample of n inputs in the NARX input space (per-dim ranges from
    training trajectory min/max).

    Returns:
      ywin_coll : (n, w, n_y)  state-window samples [y(t-1), ..., y(t-w)]
      u_coll    : (n, n_u)     input sample u(t-1)
    """
    rng    = np.random.default_rng(seed)
    y_tr, u_tr = traj_train["y"], traj_train["u"]
    n_y, n_u   = y_tr.shape[1], u_tr.shape[1]

    # per-dim ranges from training data
    y_lo, y_hi = y_tr.min(axis=0), y_tr.max(axis=0)
    u_lo, u_hi = u_tr.min(axis=0), u_tr.max(axis=0)

    # LHS in (w * n_y + n_u) dimensions
    ywin_coll = np.zeros((n, w, n_y), dtype=np.float32)
    for j in range(w):
        for k in range(n_y):
            ywin_coll[:, j, k] = _lhs_1d(y_lo[k], y_hi[k], n, rng)
    u_coll = np.zeros((n, n_u), dtype=np.float32)
    for k in range(n_u):
        u_coll[:, k] = _lhs_1d(u_lo[k], u_hi[k], n, rng)
    return ywin_coll, u_coll


# ============================================================================
# Torch-native plant rhs (Eqs 6-9). Differentiable w.r.t. y.
# ============================================================================
def torch_rhs(y: torch.Tensor, u: torch.Tensor,
              p: CSTRParams) -> torch.Tensor:
    """Vectorized torch evaluation of Eqs 6-9.

    Inputs:
      y : (B, 4)   [C_A, T, T_c, h]
      u : (B, 2)   [Q_f, Q_c]
    Returns dy/dt with the same shape.
    """
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
# PI-NARX model
# ============================================================================
class PINARXModel:
    def __init__(self, n_u: int, n_y: int, hp: PINARXHparams,
                  p_plant: CSTRParams | None = None):
        self.n_u, self.n_y, self.hp = n_u, n_y, hp
        self.p_plant = p_plant or CSTRParams()
        u_dim = hp.window * n_u if hp.include_u_lags else n_u
        self.in_dim  = hp.window * n_y + u_dim
        self.out_dim = n_y
        self.x_norm = ZScore()
        self.y_norm = ZScore()
        torch.manual_seed(hp.seed)
        np.random.seed(hp.seed)
        self.net = MLP(self.in_dim, self.out_dim,
                        hidden=hp.hidden, activation=hp.activation
                        ).to(hp.device)

    # ------------------------------------------------------------------
    # Physics residual - paper Eq 5:
    #
    #   L_ODE = (1/n_res) * sum_i sum_j ( (y_hat_ij - y_ij,t-1) / dt
    #                                       - f_j(y_hat_i, u_i) )^2
    #
    # NOTE: the rhs f_j is evaluated at the NETWORK PREDICTION y_hat
    # (implicit / backward Euler), NOT at y(t-1). This makes the residual
    # a self-consistency check: the network must predict an end-state
    # whose dynamics are consistent with the change since t-1.
    #
    # The paper notes: "rescale the sampled points and compute residuals in
    # the original, non-scaled values" - so residual is in ORIGINAL units.
    # ------------------------------------------------------------------
    def _physics_residual(self, ywin_coll: torch.Tensor,
                            u_coll:    torch.Tensor) -> torch.Tensor:
        """ywin_coll: (B, w, n_y);  u_coll: (B, n_u).  Returns res: (B, n_y)."""
        w     = self.hp.window
        B     = ywin_coll.shape[0]
        device = ywin_coll.device

        # Build NARX input feature in same order as make_windows()
        feat_y = ywin_coll.reshape(B, w * self.n_y)
        if self.hp.include_u_lags:
            feat_u = u_coll.unsqueeze(1).expand(B, w, self.n_u).reshape(B, -1)
        else:
            feat_u = u_coll                                  # u(t-1) only
        feat = torch.cat([feat_y, feat_u], dim=1)

        # Normalize -> predict -> denormalize
        x_mean = torch.from_numpy(self.x_norm.mean).to(device)
        x_std  = torch.from_numpy(self.x_norm.std ).to(device)
        y_mean = torch.from_numpy(self.y_norm.mean).to(device)
        y_std  = torch.from_numpy(self.y_norm.std ).to(device)
        out_n  = self.net((feat - x_mean) / x_std)
        y_hat  = out_n * y_std + y_mean                       # original units

        # rhs is evaluated at y_HAT (NOT y_prev) per Eq 5
        y_prev = ywin_coll[:, 0, :]                            # y(t-1)
        rhs    = torch_rhs(y_hat, u_coll, self.p_plant)        # <-- y_hat here
        res    = (y_hat - y_prev) / self.hp.dt_phys - rhs      # original units

        # Optional ablations (Table 5)
        if not self.hp.phys_mass:
            res = res.clone(); res[:, 0] = 0.0; res[:, 3] = 0.0
        if not self.hp.phys_energy:
            res = res.clone(); res[:, 1] = 0.0; res[:, 2] = 0.0
        return res

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, traj_train: dict, traj_val: dict | None = None,
            verbose: bool = True) -> dict:
        hp     = self.hp
        device = hp.device

        # 1. Windowed datasets
        X_tr, Y_tr = make_windows(traj_train["u"], traj_train["y"], hp.window,
                                    include_u_lags=hp.include_u_lags)
        if traj_val is not None:
            X_va, Y_va = make_windows(traj_val["u"], traj_val["y"], hp.window,
                                        include_u_lags=hp.include_u_lags)

        # 2. Z-score normalize using TRAIN statistics
        self.x_norm.fit(X_tr); self.y_norm.fit(Y_tr)
        X_tr_n = self.x_norm.transform(X_tr)
        Y_tr_n = self.y_norm.transform(Y_tr)
        X_tr_t = torch.from_numpy(X_tr_n).to(device)
        Y_tr_t = torch.from_numpy(Y_tr_n).to(device)
        if traj_val is not None:
            X_va_t = torch.from_numpy(self.x_norm.transform(X_va)).to(device)
            Y_va_t = torch.from_numpy(self.y_norm.transform(Y_va)).to(device)

        # 3. Collocation: LHS in 10-D NARX input space, per-dim training-range
        ywin_coll_np, u_coll_np = sample_collocation_narx(
            traj_train, hp.n_collocation, w=hp.window, seed=hp.seed)
        ywin_coll = torch.from_numpy(ywin_coll_np).to(device)
        u_coll    = torch.from_numpy(u_coll_np).to(device)

        # 4. Adam phase with early stopping on data-loss val
        opt = torch.optim.Adam(self.net.parameters(),
                                lr=hp.lr_adam, weight_decay=hp.weight_decay)
        n = X_tr_t.shape[0]
        hist = {"adam_data_loss": [], "adam_phys_loss": [], "val_loss": []}
        best_val   = float("inf")
        best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        epochs_no_improve = 0
        use_es = (hp.early_stop_patience > 0) and (traj_val is not None)

        for ep in range(hp.n_epochs_adam):
            perm = torch.randperm(n, device=device)
            ep_data = 0.0; ep_phys = 0.0; n_batches = 0
            for s in range(0, n, hp.batch_size):
                idx = perm[s:s + hp.batch_size]
                xb, yb = X_tr_t[idx], Y_tr_t[idx]
                pred = self.net(xb)
                L_data = ((pred - yb) ** 2).mean()
                # Random mini-batch of collocation points (matches batch size)
                coll_idx = torch.randint(0, hp.n_collocation,
                                          (hp.batch_size,), device=device)
                res = self._physics_residual(ywin_coll[coll_idx],
                                                u_coll[coll_idx])
                L_phys = (res ** 2).mean()
                L = hp.lambda_data * L_data + hp.lambda_phys * L_phys
                opt.zero_grad(); L.backward(); opt.step()
                ep_data += float(L_data.item()); ep_phys += float(L_phys.item())
                n_batches += 1
            ep_data /= max(n_batches, 1); ep_phys /= max(n_batches, 1)
            hist["adam_data_loss"].append(ep_data)
            hist["adam_phys_loss"].append(ep_phys)

            if traj_val is not None:
                with torch.no_grad():
                    vp = self.net(X_va_t)
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
                msg = (f"  adam ep {ep:>4}/{hp.n_epochs_adam}  "
                        f"L_data={ep_data:.4g}  L_phys={ep_phys:.4g}")
                if traj_val is not None: msg += f"  val={hist['val_loss'][-1]:.4g}"
                print(msg)

            if use_es and epochs_no_improve >= hp.early_stop_patience:
                if verbose:
                    print(f"  early-stopped at ep {ep+1}")
                break

        if use_es:
            self.net.load_state_dict(best_state)
            hist["best_val_adam"] = best_val

        # 5. L-BFGS finetune (data + physics, full batch)
        if hp.lbfgs_iters > 0:
            lbfgs = torch.optim.LBFGS(self.net.parameters(),
                                       max_iter=hp.lbfgs_iters,
                                       history_size=50,
                                       tolerance_grad=1e-9,
                                       tolerance_change=1e-12,
                                       line_search_fn="strong_wolfe")
            def closure():
                lbfgs.zero_grad()
                pred = self.net(X_tr_t)
                L_data = ((pred - Y_tr_t) ** 2).mean()
                res = self._physics_residual(ywin_coll, u_coll)
                L_phys = (res ** 2).mean()
                L = hp.lambda_data * L_data + hp.lambda_phys * L_phys
                L.backward()
                return L
            final = float(lbfgs.step(closure).item())
            hist["lbfgs_final_loss"] = final
            if verbose:
                print(f"  lbfgs done    composite={final:.6g}")
                if traj_val is not None:
                    with torch.no_grad():
                        vp = self.net(X_va_t)
                        vl = float(((vp - Y_va_t) ** 2).mean().item())
                    print(f"  lbfgs final   val_data={vl:.6g}")
                    hist["val_loss_final"] = vl
        return hist

    # ------------------------------------------------------------------
    # Evaluation - reuses NARXModel rollout / eval_mae logic
    # ------------------------------------------------------------------
    def rollout(self, u: np.ndarray, y_init: np.ndarray) -> np.ndarray:
        from narx import NARXModel
        # cheap hack: build a NARXModel shim and reuse its rollout impl
        shim = NARXModel.__new__(NARXModel)
        shim.n_u, shim.n_y, shim.hp = self.n_u, self.n_y, self.hp
        shim.net = self.net
        shim.x_norm = self.x_norm
        shim.y_norm = self.y_norm
        return shim.rollout(u, y_init)

    def eval_mae(self, traj: dict) -> float:
        from narx import NARXModel
        shim = NARXModel.__new__(NARXModel)
        shim.n_u, shim.n_y, shim.hp = self.n_u, self.n_y, self.hp
        shim.net = self.net
        shim.x_norm = self.x_norm
        shim.y_norm = self.y_norm
        return shim.eval_mae(traj)

    def eval_mae_one_step(self, traj: dict) -> float:
        from narx import NARXModel
        shim = NARXModel.__new__(NARXModel)
        shim.n_u, shim.n_y, shim.hp = self.n_u, self.n_y, self.hp
        shim.net = self.net
        shim.x_norm = self.x_norm
        shim.y_norm = self.y_norm
        return shim.eval_mae_one_step(traj)


# ============================================================================
# Convenience entry point
# ============================================================================
def train_and_eval_pi_narx(traj_train, traj_val, traj_test1, traj_test2,
                             hp: PINARXHparams | None = None,
                             verbose: bool = True) -> dict:
    hp = hp or PINARXHparams()
    n_u, n_y = traj_train["u"].shape[1], traj_train["y"].shape[1]
    model = PINARXModel(n_u=n_u, n_y=n_y, hp=hp)
    if verbose:
        n_params = sum(p.numel() for p in model.net.parameters())
        print(f"PI-NARX:  in_dim={model.in_dim}  out_dim={model.out_dim}  "
                f"params={n_params:,}")
    hist = model.fit(traj_train, traj_val, verbose=verbose)
    one1 = model.eval_mae_one_step(traj_test1)
    one2 = model.eval_mae_one_step(traj_test2)
    ar1  = model.eval_mae(traj_test1)
    ar2  = model.eval_mae(traj_test2)
    if verbose:
        print(f"\n  Test 1 ONE-STEP MAE      = {one1:.6f}  "
                f"(paper PI-NARX = 0.001242)")
        print(f"  Test 2 ONE-STEP MAE      = {one2:.6f}  "
                f"(paper PI-NARX = 0.01556)")
        print(f"  Test 1 autoregress  MAE  = {ar1:.6f}")
        print(f"  Test 2 autoregress  MAE  = {ar2:.6f}")
    return {"model": model, "hist": hist,
             "test1_one_step": one1, "test2_one_step": one2,
             "test1_autoreg":  ar1,  "test2_autoreg":  ar2}


# ============================================================================
# Self-test
# ============================================================================
if __name__ == "__main__":
    import argparse, torch as _t
    from data_gen import (gen_train_val_split, gen_test1_set, gen_test2_set)

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if _t.cuda.is_available() else "cpu")
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--n-total",type=int, default=5000)
    ap.add_argument("--n-train",type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lbfgs",  type=int, default=9000)
    ap.add_argument("--lambda-data", type=float, default=1e10)
    ap.add_argument("--lambda-phys", type=float, default=1e-2)
    ap.add_argument("--n-coll",      type=int, default=10_000)
    args = ap.parse_args()

    print("=" * 70)
    print("PI-NARX (paper Table 2 target: Test1=0.001242, Test2=0.01556)")
    print(f"device = {args.device}, seed = {args.seed}, "
          f"lbfgs = {args.lbfgs}, lambda = ({args.lambda_data:g}, {args.lambda_phys:g})")
    print("=" * 70)

    print("Generating data...")
    tr, va = gen_train_val_split(N_total=args.n_total, N_train=args.n_train,
                                    seed=args.seed)
    t1 = gen_test1_set()
    t2 = gen_test2_set()
    print(f"  train={tr['u'].shape[0]}  val={va['u'].shape[0]}  "
            f"t1={t1['u'].shape[0]}  t2={t2['u'].shape[0]}")

    # Paper-faithful recipe: Adam 1000 ep + L-BFGS 9000 iters; lambda_l=1e10,
    # lambda_p=0.01; 10-D input (u(t-1) only); LHS collocation in 10-D NARX
    # input space; residual in original units evaluated at y_hat (Eq 5).
    hp = PINARXHparams(
        window=2, hidden=(200, 400, 200), activation="tanh",
        lr_adam=1e-3, n_epochs_adam=args.epochs, batch_size=64,
        lbfgs_iters=args.lbfgs, weight_decay=0.0,
        early_stop_patience=120, seed=args.seed,
        lambda_data=args.lambda_data, lambda_phys=args.lambda_phys,
        n_collocation=args.n_coll,
        dt_phys=1.0,
        include_u_lags=False,
        device=args.device,
    )

    out = train_and_eval_pi_narx(tr, va, t1, t2, hp=hp, verbose=True)
