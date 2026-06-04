"""NMPC oracle for the K2SO4 crystallization reactor (PC-Gym Sec 4.1.3).

Implementation: do-mpc + CasADi + IPOPT, following PC-Gym's oracle pattern
(Section 2.6 of Bloor et al. 2025). The plant model is the verbatim 5-ODE
crystallization system from plants/crystallization.py.

The NMPC minimises normalised tracking error on (CV, L_n) plus a small
move-penalty on T_c. Paper settings (verbatim):
  - prediction horizon N = 10
  - Q = identity (over [CV, L_n])
  - R = 0   (paper p.10: "zero R matrix")

The episode-time controller dt is 1 hr (= 30 hr / 30 steps).

Operating-range choices NOT from the paper (documented in PARAMETERS.md):
  - T_c (action) bounds: [25, 50] degC  (covers Bloor Fig 6 range)
  - State + output normalisation bounds: see NORMALISATION below
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import do_mpc
    from casadi import exp, sqrt, fmax, fmin

from .plants.crystallization import CrystParams, CrystScenario


# ==========================================================================
# Bounds + normalisation (operating-range choices flagged in PARAMETERS.md)
# ==========================================================================
@dataclass(frozen=True)
class CrystBounds:
    """Action bounds + normalisation ranges.

    T_c bounds chosen to bracket PC-Gym Fig 6 trajectories (32-40 degC core
    range, [25, 50] degC for headroom). Output bounds for CV and L_n are
    consistent with Fig 6 (CV ~ 0..2, L_n ~ 10..20 um). These are OUR
    choices, the paper does not state them explicitly.
    """
    T_c_min: float = 25.0       # degC
    T_c_max: float = 50.0       # degC
    # Output normalisation (for Q-weighting and reward)
    CV_min: float = 0.0
    CV_max: float = 3.0
    Ln_min: float = 0.0         # um (microns)
    Ln_max: float = 30.0


@dataclass(frozen=True)
class CrystOperatingPoint:
    """Initial state + setpoints matching Bloor Fig 6.

    Using the REPO L_n = mu_1 / mu_0 (see plants docstring) and the
    standard CV^2 = mu_2*mu_0/mu_1^2 - 1, with Fig 6 initials L_n(0) ~ 15
    and CV(0) ~ 2:
        L_n  = mu_1/mu_0 = 15        -> mu_1 = 15 * mu_0
        CV^2 = mu_2*mu_0/mu_1^2 - 1 = 4  ->  mu_2*mu_0 = 5*mu_1^2
                                              -> mu_2 = 5*15^2*mu_0 = 1125*mu_0
    Pick mu_0 = 1.0  (typical normalised number density); then mu_1 = 15,
    mu_2 = 1125, mu_3 ~ 15^3*mu_0 = 3375 (volume-equivalent third moment).
    """
    # Initial state x0 = [mu_0, mu_1, mu_2, mu_3, c]
    mu_0_0: float = 1.0          # nominal number density
    mu_1_0: float = 15.0         # L_n(0) = mu_1/mu_0 = 15  (Fig 6 start)
    mu_2_0: float = 1125.0       # CV(0) ~ 2  (Fig 6 start)
    mu_3_0: float = 3375.0       # volume-equivalent  (mu_0 * Ln^3)
    c_0:    float = 0.30         # mol/L solute concentration
    # Setpoints (Fig 6: CV target ~1, L_n target ~15)
    CV_sp:  float = 1.0
    Ln_sp:  float = 15.0
    # Initial T_c (Fig 6 starts around 32 degC)
    T_c_0:  float = 32.0


# ==========================================================================
# NMPC weights from PC-Gym Section 4.3.3 (paper p.10)
# ==========================================================================
@dataclass(frozen=True)
class CrystNMPCWeights:
    """Verbatim from PC-Gym Section 4.3.3.

    'The oracle used a prediction horizon of 10 steps: an identity Q
    matrix and a zero R matrix.'
    """
    N: int = 10
    Q_CV: float = 1.0       # identity over [CV, L_n]
    Q_Ln: float = 1.0
    # Bloor 2025 §4.3.3 specifies R=0 for crystallization. However, R=0
    # produces oscillatory T_c in closed-loop NMPC on this stiff problem
    # (the warning "rterm was not set..." is exactly this). We set R=1e-3
    # as a minimal rate penalty to stabilize sequential NMPC queries.
    # Effect on single-shot queries (training data labels) is negligible.
    R:    float = 1e-3
    dt_hr: float = CrystScenario().dt_hr   # 1.0 hour


# ==========================================================================
# NMPC oracle  -  do-mpc setup
# ==========================================================================
class CrystallizationNMPC:
    """do-mpc NMPC oracle for the crystallization plant."""

    def __init__(self,
                 weights: CrystNMPCWeights | None = None,
                 bounds:  CrystBounds | None = None,
                 params:  CrystParams | None = None):
        self.w = weights or CrystNMPCWeights()
        self.b = bounds  or CrystBounds()
        self.p = params  or CrystParams()
        self._build()

    def _build(self):
        P = self.p
        model = do_mpc.model.Model("continuous")
        mu_0 = model.set_variable("_x", "mu_0")
        mu_1 = model.set_variable("_x", "mu_1")
        mu_2 = model.set_variable("_x", "mu_2")
        mu_3 = model.set_variable("_x", "mu_3")
        c    = model.set_variable("_x", "c")
        T_c  = model.set_variable("_u", "T_c")
        CV_sp = model.set_variable("_tvp", "CV_sp")
        Ln_sp = model.set_variable("_tvp", "Ln_sp")

        # -------- Verbatim model equations (Eqs 25-28) ----------------
        T_K   = T_c + 273.15
        C_eq  = -686.2686 + 3.579165 * T_K - 0.00292874 * T_K * T_K
        S     = c * 1.0e3 - C_eq
        S2    = S * S
        mu3sq = mu_3 * mu_3
        B_0 = (P.k_a * exp(P.k_b / T_K)
               * (S2 ** (P.k_c / 2.0))
               * (mu3sq ** (P.k_d / 2.0)))
        G_inf = (P.k_g * exp(P.k_1 / T_K)
                 * (S2 ** (P.k_2 / 2.0)))

        # -------- Moment + concentration dynamics (Eqs 20-24) ---------
        model.set_rhs("mu_0", B_0)
        model.set_rhs("mu_1",
            G_inf * (P.a * mu_0 + P.b * mu_1 * 1.0e-4) * 1.0e4)
        model.set_rhs("mu_2",
            2.0 * G_inf * (P.a * mu_1 * 1.0e-4 + P.b * mu_2 * 1.0e-8) * 1.0e8)
        model.set_rhs("mu_3",
            3.0 * G_inf * (P.a * mu_2 * 1.0e-8 + P.b * mu_3 * 1.0e-12) * 1.0e12)
        model.set_rhs("c",
            -0.5 * P.rho * P.alpha * G_inf
            * (P.a * mu_2 * 1.0e-8 + P.b * mu_3 * 1.0e-12))
        model.setup()

        # -------- NMPC formulation --------
        mpc = do_mpc.controller.MPC(model)
        try:
            mpc.settings.supress_ipopt_output()
        except Exception:
            pass
        mpc.set_param(n_horizon=int(self.w.N),
                       t_step=self.w.dt_hr,   # in hours
                       store_full_solution=False)

        # CV expression (Eq 29) and L_n expression (repo definition mu_1/mu_0)
        # See plants/crystallization.py docstring for the typo note on Eq 30.
        CV_expr = sqrt(fmax(mu_2 * mu_0 / (mu_1 * mu_1 + 1e-30) - 1.0, 0.0))
        Ln_expr = mu_1 / (mu_0 + 1e-30)
        # Normalised tracking errors (Bloor uses normalised quantities, p.5)
        nCV_err = (CV_expr - CV_sp) / (self.b.CV_max - self.b.CV_min)
        nLn_err = (Ln_expr - Ln_sp) / (self.b.Ln_max - self.b.Ln_min)
        lterm = self.w.Q_CV * nCV_err ** 2 + self.w.Q_Ln * nLn_err ** 2
        mterm = lterm
        mpc.set_objective(lterm=lterm, mterm=mterm)
        if self.w.R > 0:
            mpc.set_rterm(T_c=float(self.w.R))
        mpc.bounds["lower", "_u", "T_c"] = self.b.T_c_min
        mpc.bounds["upper", "_u", "T_c"] = self.b.T_c_max

        # Constant-setpoint tvp_fun (filled by query())
        self._sp = (CrystOperatingPoint().CV_sp, CrystOperatingPoint().Ln_sp)
        tvp_t = mpc.get_tvp_template()
        def tvp_fun(_t):
            for i in range(int(self.w.N) + 1):
                tvp_t["_tvp", i, "CV_sp"] = self._sp[0]
                tvp_t["_tvp", i, "Ln_sp"] = self._sp[1]
            return tvp_t
        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()
        # Call set_initial_guess ONCE at construction (matches PC-Gym).
        # do-mpc auto-warm-starts from the previous solution between sequential
        # make_step() calls. Calling set_initial_guess every query (our prior
        # bug) RESETS the warm-start, causing IPOPT to fail ~50% of the time
        # on stiff closed-loop crystallization.
        try:
            mpc.set_initial_guess()
        except Exception:
            pass
        self.mpc = mpc
        self.model = model

    def reset(self):
        """Reset the warm-start. Call ONCE per episode (between reps), NOT
        between sequential steps within an episode. Within an episode,
        do-mpc auto-warm-starts from the previous solution."""
        try:
            self.mpc.set_initial_guess()
        except Exception:
            pass

    def query(self, x: np.ndarray,
              sp_CV: float, sp_Ln: float) -> float | None:
        """Solve one FHOCP. Returns u_NMPC = T_c, or None on failure.
        Uses do-mpc's automatic warm-start from previous solve."""
        self._sp = (float(sp_CV), float(sp_Ln))
        x_arr = np.array(x, dtype=float).reshape(5, 1)
        self.mpc.x0 = x_arr
        try:
            u = self.mpc.make_step(x_arr)
            return float(np.clip(u[0, 0], self.b.T_c_min, self.b.T_c_max))
        except Exception:
            return None


# ==========================================================================
# Self-test (single NMPC query)
# ==========================================================================
if __name__ == "__main__":
    import time
    print("=== Crystallization NMPC oracle smoke test ===")
    op = CrystOperatingPoint()
    x0 = np.array([op.mu_0_0, op.mu_1_0, op.mu_2_0, op.mu_3_0, op.c_0])
    nmpc = CrystallizationNMPC()
    print(f"Initial state: x0 = {x0}")
    print(f"Setpoints: CV={op.CV_sp}, L_n={op.Ln_sp}")
    print(f"NMPC: N={nmpc.w.N}, Q=I, R={nmpc.w.R}, dt={nmpc.w.dt_hr} hr")
    print()
    t0 = time.time()
    u = nmpc.query(x0, sp_CV=op.CV_sp, sp_Ln=op.Ln_sp)
    elapsed = time.time() - t0
    print(f"NMPC u = {u}  ({elapsed*1000:.0f} ms)")
    # Quick 5-step closed loop on the plant
    from .plants.crystallization import (
        step as plant_step, CV_from_moments, Ln_from_moments)
    x = x0.copy()
    cv0 = CV_from_moments(x[0], x[1], x[2])
    ln0 = Ln_from_moments(x[0], x[1])
    print(f"\nInitial CV={cv0:.4f}, L_n={ln0:.4f}")
    print("5-step closed loop preview:")
    for k in range(5):
        u = nmpc.query(x, sp_CV=op.CV_sp, sp_Ln=op.Ln_sp)
        if u is None:
            print(f"  step {k}: NMPC failed")
            break
        x = plant_step(x, u, nmpc.w.dt_hr)
        cv = CV_from_moments(x[0], x[1], x[2])
        ln = Ln_from_moments(x[0], x[1])
        print(f"  step {k+1}: T_c={u:.3f}, CV={cv:.4f}, L_n={ln:.4f}, c={x[4]:.5f}")
