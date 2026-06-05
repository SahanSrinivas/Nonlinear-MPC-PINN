"""Shared NN utilities for Res-Phys NARX: windowing, normalization, MLP.

Extracted from the original narx.py so the production code path
(plant.py -> data_gen.py -> nn_utils.py -> resphys_narx.py) is independent
of the archived NARX/PI-NARX reference implementations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# NARX-style windowed dataset builder
# ============================================================================
def make_windows(u: np.ndarray, y: np.ndarray, w: int,
                  include_u_lags: bool = False
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Build NARX inputs/targets from one trajectory.

    Inputs:
      u : (N,   n_u)   inputs at each step
      y : (N+1, n_y)   states (y[0] is the IC)
      w : window size
      include_u_lags : if True, append u(t-1), ..., u(t-w) to the feature.
                       If False (default), only u(t-1) once - matches the
                       Thosar 2025 NARX input shape.

    Outputs:
      X : (N - w + 1, w*n_y + n_u_window)  feature matrix
      Y : (N - w + 1, n_y)                  targets = y(t)
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
        ywin = [y[t - i] for i in range(1, w + 1)]
        X[j, :w * n_y] = np.concatenate(ywin)
        if include_u_lags:
            uwin = [u[t - i] for i in range(1, w + 1)]
            X[j, w * n_y:] = np.concatenate(uwin)
        else:
            X[j, w * n_y:] = u[t - 1]
        Y[j] = y[t]
    return X, Y


# ============================================================================
# MinMax normalizer to [-1, 1]
# ============================================================================
@dataclass
class MinMaxNorm:
    """Maps each channel of `arr` from [lo, hi] -> [-1, 1] componentwise.

    Exposes `.mean = (lo+hi)/2` and `.std = (hi-lo)/2` so call-sites that
    use `(x - mean) / std` produce exactly `2*(x-lo)/(hi-lo) - 1`.
    """
    lo:   np.ndarray = field(default=None)
    hi:   np.ndarray = field(default=None)
    mean: np.ndarray = field(default=None)
    std:  np.ndarray = field(default=None)

    def fit(self, arr: np.ndarray) -> "MinMaxNorm":
        self.lo = arr.min(axis=0).astype(np.float32)
        self.hi = arr.max(axis=0).astype(np.float32)
        flat = (self.hi - self.lo) < 1e-12
        if flat.any():
            self.hi = np.where(flat, self.lo + 1.0, self.hi).astype(np.float32)
        self.mean = ((self.lo + self.hi) / 2.0).astype(np.float32)
        self.std  = ((self.hi - self.lo) / 2.0).astype(np.float32)
        return self

    def fit_to_bounds(self, lo, hi) -> "MinMaxNorm":
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
    """Fully connected feedforward net; identity (linear) output."""

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
