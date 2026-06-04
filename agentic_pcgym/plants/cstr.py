"""Continuously Stirred Tank Reactor (CSTR) - PC-Gym Case Study 1.

Verbatim port of equations (14)-(15) and Table 1 parameters from:

  Bloor et al. (2025) "PC-Gym: Benchmark environments for process control
  problems." Computers and Chemical Engineering 204, 109363.
  Section 4.1.1 (Continuously Stirred Tank Reactor).

PARAMETER VERIFICATION (vs PC-Gym source model_classes.py::cstr):
  q          = 100      ✓ Paper Table 1: 100 m³/s
  V          = 100      ✓ Paper Table 1: 100 m³
  rho        = 1000     ✓ Paper Table 1: 1000 kg/m³
  C_p        = 0.239    ✓ Paper Table 1: 0.239 J/kg-K (PC-Gym uses 'C')
  deltaHr    = -5e4     ✓ Paper Table 1: -5×10⁴ J/mol
  EA_over_R  = 8750     ✓ Paper Table 1: 8750 K
  k0         = 7.2e10   ← PC-Gym repo (NOT in paper Table 1; verified from
                          src/pcgym/model_classes.py)
  UA         = 5e4      ✓ Paper Table 1: 5×10⁴ W/K
  T_f        = 350      ✓ Paper Table 1: 350 K (PC-Gym uses 'Ti')
  C_Af       = 1        ✓ Paper Table 1: 1 mol/m³ (PC-Gym uses 'Caf')

NO ASSUMPTIONS. All values match PC-Gym source code exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


# ==========================================================================
# Parameters - VERBATIM from PC-Gym Table 1 + src code
# ==========================================================================
@dataclass(frozen=True)
class CSTRParams:
    """CSTR model parameters - PC-Gym repo values, exactly."""
    q:          float = 100.0       # Inlet flowrate (m³/s)
    V:          float = 100.0       # Reactor volume (m³)
    rho:        float = 1000.0      # Density (kg/m³)
    C_p:        float = 0.239       # Specific heat (J/kg-K) [PC-Gym: C]
    deltaHr:    float = -5.0e4      # Heat of reaction (J/mol)
    EA_over_R:  float = 8750.0      # Activation energy / R (K)
    k0:         float = 7.2e10      # Pre-exponential factor (1/s) [PC-Gym]
    UA:         float = 5.0e4       # Heat transfer coefficient (W/K)
    T_f:        float = 350.0       # Inlet temperature (K) [PC-Gym: Ti]
    C_Af:       float = 1.0         # Inlet concentration (mol/m³) [PC-Gym: Caf]


# ==========================================================================
# RHS of the 2-state CSTR system  -  Eqs. (14)-(15)
# ==========================================================================
def rhs(t: float, x: np.ndarray, T_c: float,
        p: CSTRParams | None = None) -> np.ndarray:
    """CSTR ODE: x = [C_A, T], u = T_c.

    Eq. (14):  dC_A/dt = q/V * (C_Af - C_A) - r_A
    Eq. (15):  dT/dt   = q/V * (T_f - T)
                          + (-ΔH_R)*r_A / (ρ·C_p)
                          + UA*(T_c - T) / (ρ·C_p·V)

    where r_A = k0 * exp(-E_A/(R·T)) * C_A

    Inputs:
      x   (2,) array [C_A, T]
      T_c float, cooling water temperature (K), bounded 295 ≤ T_c ≤ 302

    Returns dx/dt as (2,) numpy array.
    """
    p = p or CSTRParams()
    C_A = float(x[0])
    T   = float(x[1])
    T_c = float(T_c)

    # Reaction rate (Arrhenius)
    rA = p.k0 * math.exp(-p.EA_over_R / max(T, 1e-6)) * C_A

    # Mass balance (Eq 14)
    dC_A_dt = (p.q / p.V) * (p.C_Af - C_A) - rA

    # Energy balance (Eq 15)
    dT_dt = ((p.q / p.V) * (p.T_f - T)
             + (-p.deltaHr) * rA / (p.rho * p.C_p)
             + p.UA * (T_c - T) / (p.rho * p.C_p * p.V))

    return np.array([dC_A_dt, dT_dt])


# ==========================================================================
# Plant integrator  -  one control sample
# ==========================================================================
def step(x: np.ndarray, T_c: float, dt: float,
         p: CSTRParams | None = None) -> np.ndarray:
    """Advance the 2-state x by `dt` (CSTR time units, min) under constant T_c.

    Uses scipy RK45 with tolerances matching PC-Gym's choice (rtol=1e-8,
    atol=1e-8) to avoid the LSODA infinite-loop failure on stiff segments.

    Inputs:
      x   (2,) array [C_A, T]
      T_c float, in Kelvin (295 ≤ T_c ≤ 302)
      dt  float, integration interval (min; paper uses 25/60 min per step)
    """
    p = p or CSTRParams()
    try:
        sol = solve_ivp(rhs, t_span=(0.0, dt), y0=x, args=(float(T_c), p),
                         method="RK45", rtol=1e-8, atol=1e-8,
                         max_step=dt / 20.0)
        y_last = np.array(sol.y[:, -1])
    except Exception:
        y_last = x.copy()
    if not np.all(np.isfinite(y_last)):
        y_last = np.where(np.isfinite(y_last), y_last, x)
    # Physical clamp: C_A ≥ 0 (cannot be negative)
    y_last[0] = max(y_last[0], 0.0)
    # T should also be positive; if it goes negative, something's very wrong
    y_last[1] = max(y_last[1], 1.0)
    return y_last


# ==========================================================================
# Scenario constants (PC-Gym Section 4.1.1 + paper p.10)
# ==========================================================================
@dataclass(frozen=True)
class CSTRScenario:
    """Episode timing and NMPC settings - PC-Gym §4.3.1."""
    t_end_min:    float = 25.0      # Episode length: 25 min (paper §4.3.1)
    n_steps:      int   = 60        # 60 timesteps (paper §4.3.1)
    dt_min:       float = 25.0 / 60.0   # ~0.4167 min per controller sample
    nmpc_horizon: int   = 17        # NMPC prediction horizon (paper §4.3.1)


# ==========================================================================
# Self-test  -  one RHS evaluation, no full episode
# ==========================================================================
if __name__ == "__main__":
    p = CSTRParams()
    print("=== CSTR plant smoke test ===")
    print(f"Parameters: q={p.q}, V={p.V}, rho={p.rho}, C_p={p.C_p}")
    print(f"            deltaHr={p.deltaHr}, EA/R={p.EA_over_R}, k0={p.k0}")
    print(f"            UA={p.UA}, T_f={p.T_f}, C_Af={p.C_Af}")
    print()
    # Plausible operating point (Bloor Fig 3 shows C_A ~ 0.85, T ~ 325)
    x0 = np.array([0.85, 325.0])
    Tc0 = 300.0
    dx = rhs(0.0, x0, Tc0, p)
    print(f"At x0 = {x0}, T_c = {Tc0}:  dx/dt = {dx}")
    # Simulate one timestep
    dt = CSTRScenario().dt_min
    x_next = step(x0, Tc0, dt, p)
    print(f"After dt = {dt:.3f} min: x_next = {x_next}")
