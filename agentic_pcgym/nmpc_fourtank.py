"""NMPC oracle for the four-tank system (PC-Gym Sec 4.1.4).

Implementation: do-mpc + CasADi + IPOPT, following PC-Gym's oracle pattern.
The plant model is the verbatim 4-ODE system from plants/fourtank.py.

The NMPC minimises normalised tracking error on (h_1, h_2) with PC-Gym's
verbatim settings:
  - prediction horizon N = 17
  - Q = identity (over [h_1, h_2])
  - R = 0

Controller dt = 1000/60 = 16.667 s.

Operating-range choices NOT from the paper:
  - v_1, v_2 (pump voltage) bounds: [0, 10] V  (covers Bloor Fig 7 range)
  - Tank-level normalisation bounds: [0, 1.5] m for h_1, h_2, h_3, h_4
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import do_mpc
    from casadi import sqrt as ca_sqrt, fmax

from .plants.fourtank import FourTankParams, FourTankScenario


# ==========================================================================
# Bounds + normalisation (documented in PARAMETERS.md as our choices)
# ==========================================================================
@dataclass(frozen=True)
class FourTankBounds:
    """Action and state bounds.

    Computed steady-state analysis: with k_1=8.5e-4, a_1=3.5e-3, gamma_1=0.2,
    the maximum achievable h_1 at v_1=v_2=v_max is approximately
        h_1_max ~ ((gamma_1*k_1*v_max + a_3*sqrt(2g*h_3_ss)) / a_1)^2 / (2g)
    For v_max=10 -> h_1_max ~ 0.36 m (cannot reach setpoint 0.5).
    For v_max=15 -> h_1_max ~ 0.82 m (comfortably above setpoint 0.5).

    We use v_max = 15 V to make Bloor Fig 7 setpoints (~0.45, 0.6) physically
    reachable. Heights normalised on [0, 1.5] m.
    """
    v_min: float = 0.0
    v_max: float = 15.0
    h_min: float = 0.0
    h_max: float = 1.5


@dataclass(frozen=True)
class FourTankOperatingPoint:
    """Initial state + setpoints (consistent with Bloor Fig 7).

    Fig 7 shows initial (h_1, h_2) ~ (0.4, 0.2) and setpoint trajectories
    h_1: 0.4 -> 0.6 -> 0.1
    h_2: 0.2 -> 0.3 -> 0.2
    """
    h_1_0: float = 0.4
    h_2_0: float = 0.2
    h_3_0: float = 0.2
    h_4_0: float = 0.2
    v_1_0: float = 4.0
    v_2_0: float = 4.0
    h_1_sp: float = 0.5
    h_2_sp: float = 0.3


# ==========================================================================
# NMPC weights from PC-Gym Section 4.3.4 (paper p.10)
# ==========================================================================
@dataclass(frozen=True)
class FourTankNMPCWeights:
    """Verbatim from PC-Gym Section 4.3.4.

    'The oracle MPC used a prediction horizon of 17 steps, an identity Q
    matrix, and a zero R matrix.'
    """
    N: int = 17
    Q_h1: float = 1.0       # identity over [h_1, h_2]
    Q_h2: float = 1.0
    R:    float = 0.0       # zero R matrix
    dt_s: float = FourTankScenario().dt_s     # 1000/60 = 16.667 s


# ==========================================================================
# NMPC oracle
# ==========================================================================
class FourTankNMPC:
    """do-mpc NMPC oracle for the four-tank plant."""

    def __init__(self,
                 weights: FourTankNMPCWeights | None = None,
                 bounds:  FourTankBounds | None = None,
                 params:  FourTankParams | None = None):
        self.w = weights or FourTankNMPCWeights()
        self.b = bounds  or FourTankBounds()
        self.p = params  or FourTankParams()
        self._build()

    def _build(self):
        P = self.p
        model = do_mpc.model.Model("continuous")
        h_1 = model.set_variable("_x", "h_1")
        h_2 = model.set_variable("_x", "h_2")
        h_3 = model.set_variable("_x", "h_3")
        h_4 = model.set_variable("_x", "h_4")
        v_1 = model.set_variable("_u", "v_1")
        v_2 = model.set_variable("_u", "v_2")
        h1_sp = model.set_variable("_tvp", "h1_sp")
        h2_sp = model.set_variable("_tvp", "h2_sp")

        # ------- Verbatim plant equations (Eqs 31-34) ---------------
        # Guard with sqrt(h + eps) to keep the gradient finite at h=0
        # (raw sqrt has infinite gradient at 0 -> NaN in Jacobian).
        twog = 2.0 * P.g_a
        eps_h = 1.0e-6
        s1 = ca_sqrt(fmax(h_1, 0.0) * twog + eps_h)
        s2 = ca_sqrt(fmax(h_2, 0.0) * twog + eps_h)
        s3 = ca_sqrt(fmax(h_3, 0.0) * twog + eps_h)
        s4 = ca_sqrt(fmax(h_4, 0.0) * twog + eps_h)
        model.set_rhs("h_1",
            -(P.a_1 / P.A_1) * s1
            + (P.a_3 / P.A_1) * s3
            + (P.gamma_1 * P.k_1 / P.A_1) * v_1)
        model.set_rhs("h_2",
            -(P.a_2 / P.A_2) * s2
            + (P.a_4 / P.A_2) * s4
            + (P.gamma_2 * P.k_2 / P.A_2) * v_2)
        model.set_rhs("h_3",
            -(P.a_3 / P.A_3) * s3
            + ((1.0 - P.gamma_2) * P.k_2 / P.A_3) * v_2)
        model.set_rhs("h_4",
            -(P.a_4 / P.A_4) * s4
            + ((1.0 - P.gamma_1) * P.k_1 / P.A_4) * v_1)
        model.setup()

        # ------- NMPC formulation -----------
        mpc = do_mpc.controller.MPC(model)
        try:
            mpc.settings.supress_ipopt_output()
        except Exception:
            pass
        mpc.set_param(n_horizon=int(self.w.N),
                       t_step=self.w.dt_s,
                       store_full_solution=False)

        # Normalised tracking error (consistent with Bloor's normalised reward)
        h_range = self.b.h_max - self.b.h_min
        n_h1_err = (h_1 - h1_sp) / h_range
        n_h2_err = (h_2 - h2_sp) / h_range
        lterm = self.w.Q_h1 * n_h1_err ** 2 + self.w.Q_h2 * n_h2_err ** 2
        mterm = lterm
        mpc.set_objective(lterm=lterm, mterm=mterm)
        if self.w.R > 0:
            mpc.set_rterm(v_1=float(self.w.R), v_2=float(self.w.R))
        mpc.bounds["lower", "_u", "v_1"] = self.b.v_min
        mpc.bounds["upper", "_u", "v_1"] = self.b.v_max
        mpc.bounds["lower", "_u", "v_2"] = self.b.v_min
        mpc.bounds["upper", "_u", "v_2"] = self.b.v_max

        self._sp = (FourTankOperatingPoint().h_1_sp,
                     FourTankOperatingPoint().h_2_sp)
        tvp_t = mpc.get_tvp_template()
        def tvp_fun(_t):
            for i in range(int(self.w.N) + 1):
                tvp_t["_tvp", i, "h1_sp"] = self._sp[0]
                tvp_t["_tvp", i, "h2_sp"] = self._sp[1]
            return tvp_t
        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()
        self.mpc = mpc
        self.model = model

    def query(self, x: np.ndarray,
              sp_h1: float, sp_h2: float,
              u_warm: tuple = None) -> np.ndarray | None:
        """Solve one FHOCP. Returns u_NMPC = [v_1, v_2] (2,), or None."""
        self._sp = (float(sp_h1), float(sp_h2))
        x_arr = np.array(x, dtype=float).reshape(4, 1)
        self.mpc.x0 = x_arr
        # Warm-start at the mid-point of the bounds (or user override)
        u_mid = 0.5 * (self.b.v_min + self.b.v_max)
        if u_warm is None:
            u_warm = (u_mid, u_mid)
        try:
            self.mpc.u0 = np.array([[u_warm[0]], [u_warm[1]]])
            self.mpc.set_initial_guess()
            u = self.mpc.make_step(x_arr)
            return np.clip(np.array([u[0, 0], u[1, 0]], dtype=float),
                            self.b.v_min, self.b.v_max)
        except Exception:
            return None


# ==========================================================================
# Self-test
# ==========================================================================
if __name__ == "__main__":
    import time
    print("=== Four-tank NMPC oracle smoke test ===")
    op = FourTankOperatingPoint()
    x0 = np.array([op.h_1_0, op.h_2_0, op.h_3_0, op.h_4_0])
    nmpc = FourTankNMPC()
    print(f"Initial state: {x0}")
    print(f"Setpoints: h_1={op.h_1_sp}, h_2={op.h_2_sp}")
    print(f"NMPC: N={nmpc.w.N}, Q=I, R={nmpc.w.R}, dt={nmpc.w.dt_s:.2f} s")
    print()
    t0 = time.time()
    u = nmpc.query(x0, sp_h1=op.h_1_sp, sp_h2=op.h_2_sp)
    elapsed = time.time() - t0
    print(f"NMPC u = {u}  ({elapsed*1000:.0f} ms)")
    # Quick 20-step closed loop
    from .plants.fourtank import step as plant_step
    x = x0.copy()
    u_warm = (10.0, 10.0)
    print("\n20-step closed loop preview:")
    for k in range(20):
        u = nmpc.query(x, sp_h1=op.h_1_sp, sp_h2=op.h_2_sp,
                        u_warm=u_warm)
        if u is None:
            print(f"  step {k}: NMPC failed")
            break
        u_warm = (float(u[0]), float(u[1]))
        x = plant_step(x, u, nmpc.w.dt_s)
        if k < 5 or k == 19 or (k+1) % 5 == 0:
            print(f"  step {k+1}: v=({u[0]:.3f},{u[1]:.3f})  "
                  f"h_1={x[0]:.4f}  h_2={x[1]:.4f}  "
                  f"h_3={x[2]:.4f}  h_4={x[3]:.4f}")
