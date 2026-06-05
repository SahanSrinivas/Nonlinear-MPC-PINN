"""NARX baseline (paper Eqs 2-3 + §3.1).

NARX model:                                                              (Eq 2,3)

  y_hat(t) = f_theta( y(t-1), ..., y(t-w),  u(t-1), ..., u(t-w) )

with f_theta a fully-connected MLP. The paper uses:
  - window w = 2
  - hidden layers = (200, 400, 200) with tanh
  - Adam (lr=1e-3, batch=64) followed by L-BFGS finetune
  - per-channel z-score normalization of inputs and outputs

This module:
  - Builds the windowed (X, Y) pairs from a (u, y) trajectory.
  - Defines the MLP.
  - Trains with Adam then L-BFGS.
  - Evaluates the model autoregressively (closed-loop simulation) on
    Test Case 1 and Test Case 2 and returns MAE in the original units.

Loss reported in paper Tables 2, 4, 5, 6: MAE on the AUTOREGRESSIVE
multi-step prediction over the full test trajectory (sample-mean over
N test steps and 4 output channels), measured in original (denormalized)
units.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Hyperparameters (paper choices - all overridable from LLM-AutoOpt later)
# ============================================================================
@dataclass
class NARXHparams:
    window:     int          = 2
    hidden:     tuple        = (200, 400, 200)
    activation: str          = "tanh"      # tanh | relu | gelu | silu
    lr_adam:    float        = 1e-3        # paper §4.1
    lr_lbfgs:   float        = 0.1         # paper §4.1: "with a learning rate of 0.1"
    n_epochs_adam: int       = 1000        # paper: Adam for 1000 epochs
    batch_size: int          = 64
    lbfgs_iters: int         = 1000        # paper: L-BFGS for 1000 iter (NARX)
    weight_decay: float      = 0.0
    early_stop_patience: int = 100         # 0 disables early stopping
    # The paper's input vector is y(t-1), y(t-2), ..., y(t-w),  u(t-1) ONLY -
    # 10 dims total for w=2, 4 outputs, 2 inputs. We default to that. Setting
    # `include_u_lags=True` keeps the full u(t-1), ..., u(t-w) window for
    # ablations / LLM-AutoOpt experiments.
    include_u_lags: bool     = False
    seed:       int          = 0
    device:     str          = "cpu"


# ============================================================================
# Windowed dataset builder
# ============================================================================
def make_windows(u: np.ndarray, y: np.ndarray, w: int,
                  include_u_lags: bool = False
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Build NARX inputs/targets from one trajectory.

    Inputs:
      u : (N,   n_u)   inputs at each step
      y : (N+1, n_y)   states (y[0] is the IC)
      w : window size
      include_u_lags : if True, append u(t-1), u(t-2), ..., u(t-w)
                       to the feature; if False (paper default), only
                       append u(t-1) once. Paper input dim = w*n_y + n_u.

    Outputs:
      X : (N - w + 1, w*n_y + n_u_window) feature matrix
      Y : (N - w + 1, n_y)                 targets = y(t)
    """
    N = u.shape[0]
    n_y, n_u = y.shape[1], u.shape[1]
    M = N - w + 1
    if M <= 0:
        raise ValueError(f"trajectory too short for window w={w}")
    u_dim = w * n_u if include_u_lags else n_u
    X = np.zeros((M, w * n_y + u_dim), dtype=np.float32)
    Y = np.zeros((M, n_y),               dtype=np.float32)
    for j, t in enumerate(range(w, N + 1)):
        ywin = [y[t - i] for i in range(1, w + 1)]    # y(t-1), ..., y(t-w)
        X[j, :w * n_y] = np.concatenate(ywin)
        if include_u_lags:
            uwin = [u[t - i] for i in range(1, w + 1)]
            X[j, w * n_y:] = np.concatenate(uwin)
        else:
            X[j, w * n_y:] = u[t - 1]                 # u(t-1) only (paper)
        Y[j] = y[t]
    return X, Y


# ============================================================================
# Z-score normalizer (kept for back-compat / experimentation)
# ============================================================================
@dataclass
class ZScore:
    mean: np.ndarray = field(default=None)
    std:  np.ndarray = field(default=None)

    def fit(self, arr: np.ndarray) -> "ZScore":
        self.mean = arr.mean(axis=0).astype(np.float32)
        self.std  = arr.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-12] = 1.0
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return ((arr - self.mean) / self.std).astype(np.float32)

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return (arr * self.std + self.mean).astype(np.float32)


# ============================================================================
# MinMax normalizer to [-1, 1]  (paper §4: "Training outputs are first
# normalized in the range of [-1, 1]")
# ============================================================================
@dataclass
class MinMaxNorm:
    """Maps each channel of `arr` from [lo, hi] -> [-1, 1] componentwise.

    Exposes `.mean` and `.std` aliases so call-sites that were written for
    the ZScore interface keep working:
        mean = (lo + hi) / 2
        std  = (hi - lo) / 2
    Then the transform `(x - mean) / std` produces exactly the same
    `2*(x-lo)/(hi-lo) - 1` mapping, with the same broadcasting semantics
    used by ZScore.
    """
    lo:   np.ndarray = field(default=None)
    hi:   np.ndarray = field(default=None)
    mean: np.ndarray = field(default=None)
    std:  np.ndarray = field(default=None)

    def fit(self, arr: np.ndarray) -> "MinMaxNorm":
        self.lo = arr.min(axis=0).astype(np.float32)
        self.hi = arr.max(axis=0).astype(np.float32)
        # Guard against degenerate channels (hi == lo)
        flat = (self.hi - self.lo) < 1e-12
        if flat.any():
            self.hi = np.where(flat, self.lo + 1.0, self.hi).astype(np.float32)
        self.mean = ((self.lo + self.hi) / 2.0).astype(np.float32)
        self.std  = ((self.hi - self.lo) / 2.0).astype(np.float32)
        return self

    def fit_to_bounds(self, lo: np.ndarray, hi: np.ndarray) -> "MinMaxNorm":
        """Use externally-provided bounds (e.g. theoretical operating range)
        instead of empirical training min/max."""
        self.lo = np.asarray(lo, dtype=np.float32)
        self.hi = np.asarray(hi, dtype=np.float32)
        self.mean = ((self.lo + self.hi) / 2.0).astype(np.float32)
        self.std  = ((self.hi - self.lo) / 2.0).astype(np.float32)
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return ((arr - self.mean) / self.std).astype(np.float32)

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return (arr * self.std + self.mean).astype(np.float32)


# ============================================================================
# MLP
# ============================================================================
_ACT = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    """Fully connected feedforward net; identity output activation.

    Used as the f_theta in NARX (Eq 3) and as the parametric mapping in PI-NARX.
    """
    def __init__(self, in_dim: int, out_dim: int,
                 hidden: tuple = (200, 400, 200), activation: str = "tanh"):
        super().__init__()
        if activation not in _ACT:
            raise ValueError(f"unknown activation {activation!r}; "
                              f"choose from {list(_ACT)}")
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), _ACT[activation]()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# NARX wrapper - holds the MLP plus its input/output normalizers
# ============================================================================
class NARXModel:
    def __init__(self, n_u: int, n_y: int, hp: NARXHparams):
        self.n_u, self.n_y, self.hp = n_u, n_y, hp
        u_dim = hp.window * n_u if hp.include_u_lags else n_u
        self.in_dim  = hp.window * n_y + u_dim
        self.out_dim = n_y
        # Paper §4: outputs normalized to [-1, 1]. We do the same for inputs,
        # which is standard practice. Using train-set min/max.
        self.x_norm = MinMaxNorm()
        self.y_norm = MinMaxNorm()
        torch.manual_seed(hp.seed)
        np.random.seed(hp.seed)
        self.net = MLP(self.in_dim, self.out_dim,
                        hidden=hp.hidden, activation=hp.activation
                        ).to(hp.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, traj_train: dict, traj_val: dict | None = None,
            verbose: bool = True) -> dict:
        """Fit on one trajectory (training) and optionally validate on a second.

        Each `traj_*` is a dict {"u": (N, n_u), "y": (N+1, n_y), ...}.

        Returns a history dict {"adam_loss": [...], "val_loss": [...]}.
        """
        hp = self.hp

        # 1. Build windowed datasets
        X_tr, Y_tr = make_windows(traj_train["u"], traj_train["y"], hp.window,
                                    include_u_lags=hp.include_u_lags)
        if traj_val is not None:
            X_va, Y_va = make_windows(traj_val["u"], traj_val["y"], hp.window,
                                        include_u_lags=hp.include_u_lags)

        # 2. Normalize. For inputs we OVERRIDE the u dims with the paper's
        # theoretical training-input bounds Q_f in [100,140], Q_c in [10,20]
        # so that test inputs at the (100,20)/(140,10) corners normalize
        # exactly to +/-1, not to +/-1.5 because our random APRBS happened
        # to miss the corners. For y dims we use empirical train+val min/max
        # (paper trains a single 5000-min trajectory, so train+val sees the
        # full operating envelope they sampled).
        if traj_val is not None:
            X_fit = np.concatenate([X_tr, X_va], axis=0)
            Y_fit = np.concatenate([Y_tr, Y_va], axis=0)
        else:
            X_fit, Y_fit = X_tr, Y_tr
        self.x_norm.fit(X_fit); self.y_norm.fit(Y_fit)

        # Override u dims with theoretical bounds (paper §A: Q_f in [100,140],
        # Q_c in [10,20]). Format of feature: [y(t-1), ..., y(t-w), u(t-1)?]
        # so u dims start at index w*n_y.
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
        X_tr_n = self.x_norm.transform(X_tr)
        Y_tr_n = self.y_norm.transform(Y_tr)

        X_tr_t = torch.from_numpy(X_tr_n).to(hp.device)
        Y_tr_t = torch.from_numpy(Y_tr_n).to(hp.device)

        if traj_val is not None:
            X_va_t = torch.from_numpy(self.x_norm.transform(X_va)).to(hp.device)
            Y_va_t = torch.from_numpy(self.y_norm.transform(Y_va)).to(hp.device)

        # 3. Adam phase  (with optional early stopping on val loss)
        opt = torch.optim.Adam(self.net.parameters(),
                                lr=hp.lr_adam, weight_decay=hp.weight_decay)
        n = X_tr_t.shape[0]
        hist = {"adam_loss": [], "val_loss": []}
        best_val   = float("inf")
        best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        epochs_no_improve = 0
        use_es = (hp.early_stop_patience > 0) and (traj_val is not None)

        for ep in range(hp.n_epochs_adam):
            perm = torch.randperm(n, device=hp.device)
            ep_loss = 0.0; n_batches = 0
            for s in range(0, n, hp.batch_size):
                idx = perm[s:s + hp.batch_size]
                xb, yb = X_tr_t[idx], Y_tr_t[idx]
                pred = self.net(xb)
                loss = ((pred - yb) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += float(loss.item()); n_batches += 1
            ep_loss /= max(n_batches, 1)
            hist["adam_loss"].append(ep_loss)

            if traj_val is not None:
                with torch.no_grad():
                    vp = self.net(X_va_t)
                    vl = float(((vp - Y_va_t) ** 2).mean().item())
                hist["val_loss"].append(vl)

                # early stopping bookkeeping
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
                msg = f"  adam epoch {ep:>4}/{hp.n_epochs_adam} train={ep_loss:.6g}"
                if traj_val is not None: msg += f"  val={hist['val_loss'][-1]:.6g}"
                print(msg)

            if use_es and epochs_no_improve >= hp.early_stop_patience:
                if verbose:
                    print(f"  early-stopped at epoch {ep+1} "
                            f"(no val improvement for {hp.early_stop_patience} ep)")
                break

        # Restore best-by-val weights before L-BFGS
        if use_es:
            self.net.load_state_dict(best_state)
            hist["best_val_adam"] = best_val

        # 4. L-BFGS phase (full batch on training set)
        if hp.lbfgs_iters > 0:
            lbfgs = torch.optim.LBFGS(self.net.parameters(),
                                       lr=hp.lr_lbfgs,        # paper: 0.1
                                       max_iter=hp.lbfgs_iters,
                                       history_size=50,
                                       tolerance_grad=1e-9,
                                       tolerance_change=1e-12,
                                       line_search_fn="strong_wolfe")
            X_full, Y_full = X_tr_t, Y_tr_t
            def closure():
                lbfgs.zero_grad()
                pred = self.net(X_full)
                loss = ((pred - Y_full) ** 2).mean()
                loss.backward()
                return loss
            lbfgs_loss = float(lbfgs.step(closure).item())
            hist["lbfgs_final_loss"] = lbfgs_loss
            if verbose:
                print(f"  lbfgs done    train={lbfgs_loss:.6g}")
                if traj_val is not None:
                    with torch.no_grad():
                        vp = self.net(X_va_t)
                        vl = float(((vp - Y_va_t) ** 2).mean().item())
                    print(f"  lbfgs final   val={vl:.6g}")
                    hist["val_loss_final"] = vl

        return hist

    # ------------------------------------------------------------------
    # Autoregressive (closed-loop) rollout
    # ------------------------------------------------------------------
    def rollout(self, u: np.ndarray, y_init: np.ndarray) -> np.ndarray:
        """Autoregressively predict the trajectory given inputs `u` and the
        initial state window `y_init`.

        Inputs:
          u      : (N, n_u)
          y_init : (w, n_y) the first w states of the true trajectory
                    (i.e. y[0], y[1], ..., y[w-1])

        Returns:
          y_pred : (N+1, n_y)  with y_pred[:w] == y_init, then predicted.
        """
        hp = self.hp
        w  = hp.window
        N  = u.shape[0]
        if y_init.shape != (w, self.n_y):
            raise ValueError(f"y_init must be ({w}, {self.n_y}) "
                              f"got {y_init.shape}")
        y_pred = np.zeros((N + 1, self.n_y), dtype=np.float32)
        y_pred[:w] = y_init

        # We need t >= w to make a prediction (need full window)
        self.net.eval()
        with torch.no_grad():
            for t in range(w, N + 1):
                ywin = [y_pred[t - i] for i in range(1, w + 1)]
                if hp.include_u_lags:
                    uwin = [u[t - i] for i in range(1, w + 1)]
                else:
                    uwin = [u[t - 1]]
                feat = np.concatenate(ywin + uwin)[None, :]   # (1, in_dim)
                feat_n = self.x_norm.transform(feat)
                pred_n = self.net(torch.from_numpy(feat_n).to(hp.device)
                                   ).cpu().numpy()
                pred = self.y_norm.inverse(pred_n)[0]
                y_pred[t] = pred
        self.net.train()
        return y_pred

    # ------------------------------------------------------------------
    # MAE on autoregressive prediction over a test trajectory
    # ------------------------------------------------------------------
    def eval_mae(self, traj: dict, return_pred: bool = False
                  ) -> tuple[float, np.ndarray] | float:
        """Multi-step autoregressive MAE on a single test trajectory.

        Initial window seeded from the first w *true* states of the trajectory.
        Reports the mean of |y_pred - y_true| over all t >= w and all channels.

        The per-channel signal magnitudes differ (C_A ~ 1e-3, T ~ 400, etc),
        so we report MAE on Z-SCORE-NORMALIZED units to match the paper's
        convention (their Tables report MAE on the normalized outputs; all
        four channels live on the same scale of order 0.01-0.03).
        """
        u, y_true = traj["u"], traj["y"]
        w = self.hp.window
        y_pred = self.rollout(u, y_init=y_true[:w])

        # Z-normalize using the training y_norm
        y_pred_n = self.y_norm.transform(y_pred[w:])
        y_true_n = self.y_norm.transform(y_true[w:])
        mae = float(np.mean(np.abs(y_pred_n - y_true_n)))
        if return_pred:
            return mae, y_pred
        return mae

    # ------------------------------------------------------------------
    # MAE on ONE-STEP-AHEAD (teacher-forced) prediction
    # ------------------------------------------------------------------
    def eval_mae_one_step(self, traj: dict) -> float:
        """One-step-ahead MAE: at each t >= w, feed the TRUE window and
        compare the network's single-step prediction to y(t).

        Reports MAE on z-score-normalized outputs (same convention as
        eval_mae). This is the metric most papers actually report for
        NARX models.
        """
        u, y_true = traj["u"], traj["y"]
        X, Y = make_windows(u, y_true, self.hp.window,
                              include_u_lags=self.hp.include_u_lags)
        X_n = self.x_norm.transform(X)
        Y_n_true = self.y_norm.transform(Y)
        self.net.eval()
        with torch.no_grad():
            Y_n_pred = self.net(torch.from_numpy(X_n).to(self.hp.device)
                                  ).cpu().numpy()
        self.net.train()
        return float(np.mean(np.abs(Y_n_pred - Y_n_true)))


# ============================================================================
# Convenience entry point
# ============================================================================
def train_and_eval_narx(traj_train, traj_val, traj_test1, traj_test2,
                          hp: NARXHparams | None = None,
                          verbose: bool = True) -> dict:
    hp = hp or NARXHparams()
    n_u, n_y = traj_train["u"].shape[1], traj_train["y"].shape[1]
    model = NARXModel(n_u=n_u, n_y=n_y, hp=hp)
    if verbose:
        n_params = sum(p.numel() for p in model.net.parameters())
        print(f"NARX:  in_dim={model.in_dim}  out_dim={model.out_dim}  "
                f"params={n_params:,}")
    hist = model.fit(traj_train, traj_val, verbose=verbose)
    test1_mae = model.eval_mae(traj_test1)
    test2_mae = model.eval_mae(traj_test2)
    if verbose:
        print(f"\n  Test 1 (interp)        MAE = {test1_mae:.6f}")
        print(f"  Test 2 (extrapolation) MAE = {test2_mae:.6f}")
        print(f"  Paper NARX  Test 1 ~ 0.001508,  Test 2 ~ 0.01934")
    return {"model": model, "hist": hist,
             "test1_mae": test1_mae, "test2_mae": test2_mae}


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
    ap.add_argument("--lbfgs",  type=int, default=1000)
    args = ap.parse_args()

    print("=" * 70)
    print("NARX baseline (paper Table 2 target: Test1=0.001508, Test2=0.01934)")
    print(f"device = {args.device}, seed = {args.seed}")
    print("=" * 70)

    print("Generating data...")
    tr, va = gen_train_val_split(N_total=args.n_total, N_train=args.n_train,
                                    seed=args.seed)
    t1 = gen_test1_set()
    t2 = gen_test2_set()
    print(f"  train: {tr['u'].shape[0]} steps")
    print(f"  val:   {va['u'].shape[0]} steps")
    print(f"  test1: {t1['u'].shape[0]} steps")
    print(f"  test2: {t2['u'].shape[0]} steps")

    # Paper recipe: Adam(1000 ep) -> L-BFGS(1000 iter); 10-D input (u(t-1) only).
    hp = NARXHparams(
        window=2, hidden=(200, 400, 200), activation="tanh",
        lr_adam=1e-3, n_epochs_adam=args.epochs, batch_size=64,
        lbfgs_iters=args.lbfgs, weight_decay=0.0,
        early_stop_patience=120, seed=args.seed,
        include_u_lags=False,
        device=args.device,
    )

    out = train_and_eval_narx(tr, va, t1, t2, hp=hp, verbose=True)
    one1 = out["model"].eval_mae_one_step(t1)
    one2 = out["model"].eval_mae_one_step(t2)
    print(f"\n  Test 1 ONE-STEP MAE      = {one1:.6f}    "
          f"(paper NARX = 0.001508)")
    print(f"  Test 2 ONE-STEP MAE      = {one2:.6f}    "
          f"(paper NARX = 0.01934)")
