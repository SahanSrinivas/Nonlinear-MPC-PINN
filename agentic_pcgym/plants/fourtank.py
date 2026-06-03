"""Four-tank multivariable system  -  PC-Gym Case Study 4.

Verbatim port of equations (31)-(34) and Table 4 parameters from:

  Bloor et al. (2025) "PC-Gym: Benchmark environments for process control
  problems." Computers and Chemical Engineering 204, 109363.
  Section 4.1.4 (Four-tank system).

Underlying physical model from:
  Johansson (2000) IEEE Trans. Control Syst. Technol. 8, "The quadruple-
  tank process: A multivariable laboratory process with an adjustable zero."

NO ASSUMPTIONS. All values and equations exactly as printed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


# ==========================================================================
# Parameters - VERBATIM from PC-Gym Table 4 (paper p.8)
# ==========================================================================
@dataclass(frozen=True)
class FourTankParams:
    """Four-tank model parameters - PC-Gym repo values.

    Paper Table 4 lists g = 9.8 m/s^2; the repo's `model_classes.py` uses
    g = 9.81. We use the repo value (it IS the executed code).
    """
    g_a:   float = 9.81          # Paper Table 4: 9.8; repo: 9.81
    gamma_1: float = 0.20
    gamma_2: float = 0.20
    k_1:   float = 8.5e-4
    k_2:   float = 9.5e-4
    a_1:   float = 3.5e-3
    a_2:   float = 3.0e-3
    a_3:   float = 2.0e-3
    a_4:   float = 2.5e-3
    A_1:   float = 1.0
    A_2:   float = 1.0
    A_3:   float = 1.0
    A_4:   float = 1.0


# ==========================================================================
# RHS of the 4-state four-tank ODE system  -  Eqs. (31)-(34)
# ==========================================================================
def rhs(t: float, x: np.ndarray, u: np.ndarray,
        p: FourTankParams | None = None) -> np.ndarray:
    """Four-tank ODE: x = [h_1, h_2, h_3, h_4], u = [v_1, v_2].

    Eq. (31):  dh_1/dt = -(a_1/A_1)*sqrt(2*g_a*h_1)
                         + (a_3/A_1)*sqrt(2*g_a*h_3)
                         + (gamma_1*k_1/A_1)*v_1
    Eq. (32):  dh_2/dt = -(a_2/A_2)*sqrt(2*g_a*h_2)
                         + (a_4/A_2)*sqrt(2*g_a*h_4)
                         + (gamma_2*k_2/A_2)*v_2
    Eq. (33):  dh_3/dt = -(a_3/A_3)*sqrt(2*g_a*h_3)
                         + ((1-gamma_2)*k_2/A_3)*v_2
    Eq. (34):  dh_4/dt = -(a_4/A_4)*sqrt(2*g_a*h_4)
                         + ((1-gamma_1)*k_1/A_4)*v_1

    Returns dx/dt as a (4,) numpy array.
    """
    p = p or FourTankParams()
    h_1, h_2, h_3, h_4 = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    v_1, v_2 = float(u[0]), float(u[1])

    # Clamp heights at zero before sqrt (physical: levels cannot be negative)
    h_1c = max(h_1, 0.0)
    h_2c = max(h_2, 0.0)
    h_3c = max(h_3, 0.0)
    h_4c = max(h_4, 0.0)

    sqrt_2g = math.sqrt(2.0 * p.g_a)
    s1 = sqrt_2g * math.sqrt(h_1c)
    s2 = sqrt_2g * math.sqrt(h_2c)
    s3 = sqrt_2g * math.sqrt(h_3c)
    s4 = sqrt_2g * math.sqrt(h_4c)

    dh_1 = (-(p.a_1 / p.A_1) * s1
            + (p.a_3 / p.A_1) * s3
            + (p.gamma_1 * p.k_1 / p.A_1) * v_1)
    dh_2 = (-(p.a_2 / p.A_2) * s2
            + (p.a_4 / p.A_2) * s4
            + (p.gamma_2 * p.k_2 / p.A_2) * v_2)
    dh_3 = (-(p.a_3 / p.A_3) * s3
            + ((1.0 - p.gamma_2) * p.k_2 / p.A_3) * v_2)
    dh_4 = (-(p.a_4 / p.A_4) * s4
            + ((1.0 - p.gamma_1) * p.k_1 / p.A_4) * v_1)

    return np.array([dh_1, dh_2, dh_3, dh_4])


# ==========================================================================
# Plant integrator  -  one control sample
# ==========================================================================
def step(x: np.ndarray, u: np.ndarray, dt: float,
         p: FourTankParams | None = None) -> np.ndarray:
    """Advance the 4-state x by `dt` seconds under constant input u.

    Inputs:
      x   (4,) array [h_1, h_2, h_3, h_4] in meters
      u   (2,) array [v_1, v_2] pump voltages
      dt  float, integration interval in seconds
    """
    p = p or FourTankParams()
    sol = solve_ivp(rhs, t_span=(0.0, dt), y0=x,
                     args=(np.asarray(u, dtype=float), p),
                     method="LSODA", rtol=1e-9, atol=1e-12, max_step=dt)
    x_next = np.array(sol.y[:, -1])
    # Physical clamp: tank levels >= 0
    return np.clip(x_next, 0.0, None)


# ==========================================================================
# Scenario constants (PC-Gym Section 4.1.4 + paper p.10)
# ==========================================================================
@dataclass(frozen=True)
class FourTankScenario:
    """Episode timing and NMPC settings - PC-Gym paper p.10."""
    t_end_s:     float = 1000.0     # Episode length: 1000 s (paper p.10)
    n_steps:     int   = 60         # 60 timesteps (paper p.10)
    # Implied dt = 1000/60 = 16.667 s per controller sample
    dt_s:        float = 1000.0 / 60.0
    nmpc_horizon: int  = 17         # NMPC prediction horizon (paper p.10)


# ==========================================================================
# Self-test  -  one RHS evaluation, no full episode
# ==========================================================================
if __name__ == "__main__":
    p = FourTankParams()
    print("=== Four-tank plant smoke test ===")
    print(f"Parameters: g_a={p.g_a}, gamma_1={p.gamma_1}, gamma_2={p.gamma_2}")
    print(f"            k_1={p.k_1}, k_2={p.k_2}")
    print(f"            a=[{p.a_1}, {p.a_2}, {p.a_3}, {p.a_4}]")
    print(f"            A=[{p.A_1}, {p.A_2}, {p.A_3}, {p.A_4}]")
    print()
    # Plausible operating point
    x0 = np.array([0.5, 0.5, 0.3, 0.3])     # heights in m
    u0 = np.array([5.0, 5.0])               # voltages in V
    dx = rhs(0.0, x0, u0, p)
    print(f"At x0 = {x0}, u = {u0}:  dx/dt = {dx}")
    # Simulate one timestep
    dt = FourTankScenario().dt_s
    x_next = step(x0, u0, dt, p)
    print(f"After dt = {dt:.3f} s: x_next = {x_next}")
