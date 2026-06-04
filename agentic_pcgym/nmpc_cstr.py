"""NMPC oracle for the CSTR system (PC-Gym §4.1.1, §4.3.1).

do-mpc + CasADi + IPOPT, following the EXACT pattern of nmpc_fourtank.py
and the corrected warm-start pattern from nmpc_crystallization.py:
  - set_initial_guess() called ONCE at __init__ (matches PC-Gym).
  - reset(x0) method for per-episode warm-start.
  - query() does NOT call set_initial_guess (auto-warm-starts from prev solve).

NMPC settings (verbatim from Bloor §4.3.1):
  - prediction horizon N = 17
  - Q = identity (over C_A only — the controlled variable)
  - R = 0 (zero R matrix, but we add R=1e-3 for closed-loop stability,
    matching nmpc_crystallization's fix; effect on single-shot is negligible)

Operating-range choices (matching PC-Gym + Bloor):
  - T_c (action) bounds: [295, 302] K  (Bloor §4.1.1)
  - C_A normalisation: [0, 1] mol/m³   (operating range)
  - T   normalisation: [300, 400] K    (typical CSTR range)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import do_mpc
    from casadi import exp as ca_exp, fmax

from .plants.cstr import CSTRParams, CSTRScenario


# ==========================================================================
# Bounds + normalisation
# ==========================================================================
@dataclass(frozen=True)
class CSTRBounds:
    """Action and state bounds.

    Bloor §4.1.1: T_c bounded 295 ≤ T_c ≤ 302 K.
    Normalisation bounds are our choices for the NMPC's quadratic cost.
    """
    T_c_min: float = 295.0
    T_c_max: float = 302.0
    C_A_min: float = 0.0       # C_A normalisation range
    C_A_max: float = 1.0
    T_min:   float = 300.0     # T normalisation range
    T_max:   float = 400.0


@dataclass(frozen=True)
class CSTROperatingPoint:
    """Initial state + setpoint (Bloor Fig 3 operating envelope).

    Fig 3 shows C_A oscillating around 0.86, with setpoints at 0.86, 0.89.
    Initial T around 325 K.
    """
    C_A_0:  float = 0.85
    T_0:    float = 325.0
    T_c_0:  float = 300.0
    # Setpoints (Bloor Fig 3 steps: 0.86 -> 0.89 -> 0.86)
    C_A_sp: float = 0.86


# ==========================================================================
# NMPC weights
# ==========================================================================
@dataclass(frozen=True)
class CSTRNMPCWeights:
    """Verbatim from PC-Gym §4.3.1.

    'The oracle used a prediction horizon of 17 steps, an identity Q
    matrix, and a zero R matrix.' (paper p.8)

    We use R=1e-3 (small rate penalty) for closed-loop stability — see
    nmpc_crystallization.py for the same reasoning. Effect on single-shot
    NMPC queries (training data labels) is negligible.
    """
    N:    int   = 17
    Q_CA: float = 1.0          # identity over [C_A]
    R:    float = 1e-3         # small rate penalty (was 0 per paper)
    dt_min: float = CSTRScenario().dt_min   # 25/60 min


# ==========================================================================
# NMPC oracle
# ==========================================================================
class CSTRNMPC:
    """do-mpc NMPC oracle for the CSTR plant."""

    def __init__(self,
                 weights: CSTRNMPCWeights | None = None,
                 bounds:  CSTRBounds | None = None,
                 params:  CSTRParams | None = None):
        self.w = weights or CSTRNMPCWeights()
        self.b = bounds  or CSTRBounds()
        self.p = params  or CSTRParams()
        self._build()

    def _build(self):
        P = self.p
        model = do_mpc.model.Model("continuous")
        C_A   = model.set_variable("_x", "C_A")
        T     = model.set_variable("_x", "T")
        T_c   = model.set_variable("_u", "T_c")
        CA_sp = model.set_variable("_tvp", "CA_sp")

        # -------- Verbatim model equations (Eqs 14-15) ----------------
        # Reaction rate: r_A = k0 * exp(-EA/R / T) * C_A
        rA = P.k0 * ca_exp(-P.EA_over_R / fmax(T, 1e-6)) * C_A

        # Mass balance (Eq 14)
        model.set_rhs("C_A", (P.q / P.V) * (P.C_Af - C_A) - rA)
        # Energy balance (Eq 15)
        model.set_rhs("T",
            (P.q / P.V) * (P.T_f - T)
            + (-P.deltaHr) * rA / (P.rho * P.C_p)
            + P.UA * (T_c - T) / (P.rho * P.C_p * P.V))
        model.setup()

        # -------- NMPC formulation ----------
        mpc = do_mpc.controller.MPC(model)
        try:
            mpc.settings.supress_ipopt_output()
        except Exception:
            pass
        mpc.set_param(n_horizon=int(self.w.N),
                       t_step=self.w.dt_min,
                       store_full_solution=False)

        # Normalised tracking error on C_A (Bloor §2.6 uses normalised vars)
        nCA_err = (C_A - CA_sp) / (self.b.C_A_max - self.b.C_A_min)
        lterm = self.w.Q_CA * nCA_err ** 2
        mterm = lterm
        mpc.set_objective(lterm=lterm, mterm=mterm)
        if self.w.R > 0:
            mpc.set_rterm(T_c=float(self.w.R))
        mpc.bounds["lower", "_u", "T_c"] = self.b.T_c_min
        mpc.bounds["upper", "_u", "T_c"] = self.b.T_c_max

        # Constant-setpoint tvp_fun (filled by query())
        self._sp = CSTROperatingPoint().C_A_sp
        tvp_t = mpc.get_tvp_template()
        def tvp_fun(_t):
            for i in range(int(self.w.N) + 1):
                tvp_t["_tvp", i, "CA_sp"] = self._sp
            return tvp_t
        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()
        # Call set_initial_guess ONCE at construction (matches PC-Gym).
        try:
            mpc.set_initial_guess()
        except Exception:
            pass
        self.mpc = mpc
        self.model = model
        self._first_call = False

    def reset(self, x0: np.ndarray | None = None):
        """Reset warm-start. Call ONCE per episode (between reps).
        If x0 is given, sets initial state BEFORE re-initializing guess.
        """
        try:
            if x0 is not None:
                self.mpc.x0 = np.array(x0, dtype=float).reshape(2, 1)
            self.mpc.set_initial_guess()
            self._first_call = True
        except Exception:
            pass

    def query(self, x: np.ndarray, sp_CA: float) -> float | None:
        """Solve one FHOCP. Returns u_NMPC = T_c, or None on failure."""
        self._sp = float(sp_CA)
        x_arr = np.array(x, dtype=float).reshape(2, 1)
        self.mpc.x0 = x_arr
        if getattr(self, "_first_call", False):
            try:
                self.mpc.set_initial_guess()
            except Exception:
                pass
            self._first_call = False
        try:
            u = self.mpc.make_step(x_arr)
            return float(np.clip(u[0, 0], self.b.T_c_min, self.b.T_c_max))
        except Exception:
            return None


# ==========================================================================
# Self-test
# ==========================================================================
if __name__ == "__main__":
    import time
    print("=== CSTR NMPC oracle smoke test ===")
    op = CSTROperatingPoint()
    x0 = np.array([op.C_A_0, op.T_0])
    nmpc = CSTRNMPC()
    print(f"Initial state: x0 = {x0}")
    print(f"Setpoint: CA_sp = {op.C_A_sp}")
    print(f"NMPC: N={nmpc.w.N}, Q=I, R={nmpc.w.R}, dt={nmpc.w.dt_min} min")
    print()
    t0 = time.time()
    u = nmpc.query(x0, sp_CA=op.C_A_sp)
    elapsed = time.time() - t0
    print(f"NMPC u = T_c = {u:.3f} K  ({elapsed*1000:.0f} ms)")
    # 5-step closed-loop sanity check
    from .plants.cstr import step as plant_step
    x = x0.copy()
    print(f"\nStep | C_A    | T      | T_c")
    for k in range(5):
        u = nmpc.query(x, sp_CA=op.C_A_sp)
        x = plant_step(x, u, nmpc.w.dt_min)
        print(f"  {k+1}   | {x[0]:.4f} | {x[1]:.2f} | {u:.3f}")
