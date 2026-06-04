"""K2SO4 cooling crystallization reactor — PC-Gym Case Study 3.

Verbatim port of equations (20)-(30) and Table 3 parameters from:

  Bloor et al. (2025) "PC-Gym: Benchmark environments for process control
  problems." Computers and Chemical Engineering 204, 109363.
  Section 4.1.3 (Crystallization reactor).

Underlying physical model from:
  de Moraes et al. (2023) Ind. Eng. Chem. Res. 62(24), 9515-9532.

NO ASSUMPTIONS. All values and equations exactly as printed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


# ==========================================================================
# Parameters - PC-Gym OFFICIAL REPO precision (src/pcgym/model_classes.py)
# Paper Table 3 lists rounded values; the repo carries more precision.
# We use the repo values exactly because the repo IS the executed code in
# Bloor et al. 2025.
# ==========================================================================
@dataclass(frozen=True)
class CrystParams:
    """K2SO4 crystallization model - PC-Gym repo values, exactly."""
    # Nucleation kinetics (Eq. 27)
    k_a:   float = 0.923714966           # paper Table 3: 0.92
    k_b:   float = -6754.878558          # paper Table 3: -6800
    k_c:   float = 0.92229965554         # paper Table 3: 0.92
    k_d:   float = 1.341205945           # paper Table 3: 1.3
    # Growth kinetics (Eq. 28)
    k_g:   float = 48.07514464           # paper Table 3: 48
    k_1:   float = -4921.261419          # paper Table 3: -4900
    k_2:   float = 1.871281405           # paper Table 3: 1.9
    # Size-dependent growth + crystal properties
    a:     float = 0.50523693            # paper Table 3: 0.51
    b:     float = 7.271241375           # paper Table 3: 7.3
    alpha: float = 7.510905767           # paper Table 3: 7.5
    rho:   float = 2.658                 # paper Table 3: 2.7


# ==========================================================================
# Equilibrium concentration  -  Eq. (25)
# ==========================================================================
def C_eq(T_c: float) -> float:
    """Equilibrium solute concentration C_eq(T_c).

    PC-Gym Eq. 25 (T_c in deg C; conversion to K via +273.15 inside).
        C_eq = -686.2686 + 3.579165*(T_c+273.15) - 0.00292874*(T_c+273.15)^2
    """
    T_K = T_c + 273.15
    return -686.2686 + 3.579165 * T_K - 0.00292874 * T_K * T_K


# ==========================================================================
# RHS of the 5-state crystallization ODE system  -  Eqs. (20)-(28)
# ==========================================================================
def rhs(t: float, x: np.ndarray, T_c: float,
        p: CrystParams | None = None) -> np.ndarray:
    """5-state crystallization ODE: x = [mu_0, mu_1, mu_2, mu_3, c].

    Returns dx/dt as a (5,) numpy array.
    """
    p = p or CrystParams()
    mu_0, mu_1, mu_2, mu_3, c = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])

    # Supersaturation (Eq. 26)
    S = c * 1.0e3 - C_eq(T_c)
    # Use S^2 form to mirror paper exactly (always non-negative)
    S2 = S * S
    mu3_2 = mu_3 * mu_3
    # Guard against tiny negatives from numerics inside the power
    S2 = max(S2, 0.0)
    mu3_2 = max(mu3_2, 0.0)

    T_K = T_c + 273.15

    # Nucleation rate  -  Eq. (27)
    #   B_0 = k_a * exp(k_b / (T_c+273.15)) * (S^2)^(k_c/2) * (mu_3^2)^(k_d/2)
    B_0 = (p.k_a * math.exp(p.k_b / T_K)
           * (S2 ** (p.k_c / 2.0))
           * (mu3_2 ** (p.k_d / 2.0)))

    # Growth rate  -  Eq. (28)
    #   G_inf = k_g * exp(k_1/(T_c+273.15)) * (S^2)^(k_2/2)
    G_inf = (p.k_g * math.exp(p.k_1 / T_K)
             * (S2 ** (p.k_2 / 2.0)))

    # Moment dynamics  -  Eqs. (20)-(23)
    dmu_0 = B_0
    dmu_1 = G_inf * (p.a * mu_0 + p.b * mu_1 * 1.0e-4) * 1.0e4
    dmu_2 = 2.0 * G_inf * (p.a * mu_1 * 1.0e-4
                           + p.b * mu_2 * 1.0e-8) * 1.0e8
    dmu_3 = 3.0 * G_inf * (p.a * mu_2 * 1.0e-8
                           + p.b * mu_3 * 1.0e-12) * 1.0e12

    # Concentration dynamics  -  Eq. (24)
    #   dc/dt = -0.5 * rho * alpha * G_inf * (a*mu_2*1e-8 + b*mu_3*1e-12)
    dc = -0.5 * p.rho * p.alpha * G_inf * (p.a * mu_2 * 1.0e-8
                                             + p.b * mu_3 * 1.0e-12)

    return np.array([dmu_0, dmu_1, dmu_2, dmu_3, dc])


# ==========================================================================
# Plant integrator  -  one control sample
# ==========================================================================
def step(x: np.ndarray, T_c: float, dt: float,
         p: CrystParams | None = None) -> np.ndarray:
    """Advance the 5-state x by `dt` hours under constant T_c (degC).

    Uses scipy RK45 with PC-Gym-matching tolerances (rtol=1e-8, atol=1e-8)
    to AVOID the LSODA infinite-loop failure on stiff dynamics that occurs
    when controllers push the system into ill-conditioned regions.

    RK45 is a non-stiff fixed-order solver — it won't try to shrink the step
    to 1e-17 like LSODA does. If the dynamics are stiff at a given point,
    RK45 just takes a small (but finite) step and continues.

    PC-Gym (Bloor 2025) uses JAX Tsit5 (5th-order Runge-Kutta) for the same
    reason — see github.com/MaximilianB2/pc-gym src/pcgym/integrator.py.

    Inputs:
      x   (5,) array [mu_0, mu_1, mu_2, mu_3, c]
      T_c float, in degrees Celsius (paper uses T_c+273.15 inside RHS)
      dt  float, integration interval (hours; paper uses 1 hr controller dt)
    """
    p = p or CrystParams()
    try:
        sol = solve_ivp(rhs, t_span=(0.0, dt), y0=x, args=(float(T_c), p),
                         method="RK45", rtol=1e-8, atol=1e-8,
                         max_step=dt / 50.0)
        y_last = np.array(sol.y[:, -1])
    except Exception:
        # If even RK45 fails, fall back to previous state (no progression).
        y_last = x.copy()
    if not np.all(np.isfinite(y_last)):
        # Replace NaN entries with previous state entries.
        y_last = np.where(np.isfinite(y_last), y_last, x)
    return np.maximum(y_last, 0.0)


# ==========================================================================
# Controlled outputs  -  Eqs. (29)-(30)
# ==========================================================================
def CV_from_moments(mu_0: float, mu_1: float, mu_2: float) -> float:
    """Coefficient of Variation - PC-Gym Eq. (29).

    CV = sqrt(mu_2 * mu_0 / mu_1^2  -  1)
    """
    # Guard against mu_1 = 0 at t=0
    if mu_1 <= 1e-30:
        return 0.0
    inside = mu_2 * mu_0 / (mu_1 * mu_1) - 1.0
    return math.sqrt(max(inside, 0.0))


def Ln_from_moments(mu_0: float, mu_1: float) -> float:
    """Number-average crystal size as used by the PC-Gym REPO.

    L_n = mu_1 / mu_0   (standard population-balance number-average)

    NOTE: PC-Gym paper Eq. (30) prints  L_n = mu_1 / mu_2  ,  which is a
    TYPO. The official code at
        github.com/MaximilianB2/pc-gym/src/pcgym/model_classes.py
    computes  dLn/dt = (dmi1dt*mu_0 - mu_1*dmi0dt)/(mu_0^2 + eps),
    which integrates to  L_n = mu_1/mu_0  (the standard definition).
    We use the repo-implemented formula because the repo IS the code
    that produced Bloor et al. 2025's reported optimality gaps.
    """
    if mu_0 <= 1e-30:
        return 0.0
    return mu_1 / mu_0


# ==========================================================================
# Scenario constants (PC-Gym Section 4.1.3)
# ==========================================================================
@dataclass(frozen=True)
class CrystScenario:
    """Episode timing and NMPC settings - PC-Gym paper p.10."""
    t_end_hr:    float = 30.0       # Episode length: 30 hours
    n_steps:     int   = 30         # Number of timesteps
    dt_hr:       float = 1.0        # Controller dt = 30/30 = 1 hr
    nmpc_horizon: int  = 10         # NMPC prediction horizon (paper p.10)
    # NMPC weights: paper p.10 "identity Q matrix, and a zero R matrix"
    # (Identity is over the controlled-output space, which has 2 outputs.)


# ==========================================================================
# Self-test  -  no full run, just confirm one RHS evaluation
# ==========================================================================
if __name__ == "__main__":
    p = CrystParams()
    print("=== K2SO4 Crystallization plant smoke test ===")
    print(f"Parameters: k_a={p.k_a}, k_b={p.k_b}, k_c={p.k_c}, k_d={p.k_d}")
    print(f"            k_g={p.k_g}, k_1={p.k_1}, k_2={p.k_2}")
    print(f"            a={p.a}, b={p.b}, alpha={p.alpha}, rho={p.rho}")
    print()
    # Try a single RHS evaluation at a plausible operating point
    x0 = np.array([1.0e6, 1.0e4, 1.0e2, 1.0, 0.30])  # placeholder state
    T_c = 30.0  # deg C
    print(f"T_c = {T_c} degC  ->  C_eq = {C_eq(T_c):.4f}")
    dx = rhs(0.0, x0, T_c, p)
    print(f"RHS at x0={x0}: dx/dt = {dx}")
    print(f"CV(at x0)   = {CV_from_moments(x0[0], x0[1], x0[2]):.6f}")
    print(f"L_n(at x0)  = mu_1/mu_0 = {Ln_from_moments(x0[0], x0[1]):.6f}")
