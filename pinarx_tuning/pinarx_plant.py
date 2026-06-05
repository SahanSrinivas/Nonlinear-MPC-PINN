"""CSTR with cooling jacket - plant simulator for the Thosar 2025 PI-NARX paper.

Verbatim port of the system of ODEs (Eqs 6-9) and parameters (Table 1) from:

  Thosar, Bhakte, Li, Srinivasan, Prasad (2025).
  "A novel hybrid neural network for modeling dynamic systems using
   physics-informed regularization."
  J. Process Control 152, 103473.

Equations (Eqs 6-9):

  dC_A/dt = -k_0 * exp(-E_a/(R*T)) * C_A
            + (Q_f * C_Af - C_V * sqrt(h) * C_A) / (A*h)                  (6)

  dT/dt   =  k_0 * exp(-E_a/(R*T)) * C_A * (-dH) / (rho*C_P)
            + (Q_f * T_f - C_V * sqrt(h) * T) / (A*h)
            + UA_c * (T_c - T) / (rho*C_P*A*h)                            (7)

  dT_c/dt =  Q_c * (T_cf - T_c) / V_c
            + UA_c * (T - T_c) / (rho_c*C_Pc*V_c)                         (8)

  dh/dt   = (Q_f - C_V * sqrt(h)) / A                                     (9)

State:                 y = [C_A, T, T_c, h]   (mol/L, K, K, m)
Manipulated inputs:    u = [Q_f, Q_c]         (L/min, L/min)

NOTE on parameters (Table 1 vs. Appendix y_SS):
  Paper Table 1 has THREE typos. Only the values below reproduce
  y_SS = [0.0025, 416.12, 351.55, 9] given in the Appendix (we get
  max 0.19% error across all four states):

    - C_V  = 40   (not 400):    from Eq 9 at h_ss=9, Q_f=120,
                                  C_V * sqrt(9) = 120  ->  C_V = 40.

    - k_0  = 7.2e10 (not 1.0e10): CSTR mass balance at SS with V=A*h=900 L
                                    and Q_f=120 L/min needs k_eff ~ 53/min
                                    at T=416.12 K -> k_0 = 7.2e10.

    - T_cf = 300 K (not 320 K):  Fig 2 schematic actually shows 300 K;
                                  Table 1 has it as 320. Coolant SS Eq 8
                                  at (T=416.12, T_c=351.55, Q_c=15) only
                                  closes if T_cf = 300. Fig 2 wins.

  k_0/C_V match the standard Bequette CSTR textbook values (cited via
  Johannesmeyer 2002 [88] in the paper); T_cf=300 matches the figure
  in the same paper. We use all three corrected values so the whole
  reproduction matches the Appendix.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


# ============================================================================
# Parameters - Thosar 2025 Table 1
# ============================================================================
@dataclass(frozen=True)
class CSTRParams:
    """CSTR parameters. Values follow Table 1 EXCEPT where they conflict with
    the published steady-state y_SS = [0.0025, 416.12, 351.55, 9].
    """
    # Reaction kinetics (Table 1 says 1.0e10, calibrated to y_SS - see header)
    k_0:      float = 7.2e10    # 1/min   (Bequette/Johannesmeyer standard)
    Ea_over_R: float = 8750.0   # K       (Table 1, no conflict)

    # Feed conditions
    C_Af:     float = 1.0       # mol/L  (Table 1, no conflict)
    T_f:      float = 320.0     # K      (Table 1, no conflict)
    T_cf:     float = 300.0     # K      (Fig 2 shows 300; Table 1 says 320 -
                                #         coolant SS only consistent with 300)

    # Outlet hydraulics (Eq 9 implies C_V=40, not Table 1's 400 - see header)
    C_V:      float = 40.0      # L/(min * sqrt(m))  -- calibrated to y_SS

    # Energy balance terms (Table 1, no conflict)
    delta_H:  float = -5.0e4    # J/mol
    rho_Cp:   float = 239.0     # J/(L * K)        - product of rho and C_P
    rho_c_Cpc: float = 4175.0   # J/(L * K)        - coolant
    UA_c:     float = 5.0e4     # J/(min * K)

    # Vessel geometry
    A:        float = 100.0     # cross-section, m^2 (Table 1)
    V_c:      float = 250.0     # coolant volume, L (Table 1)

    # Numerical safety
    h_min:    float = 0.01      # avoid division by zero / sqrt(neg)


# ============================================================================
# Right-hand side of the 4-state ODE  -  Eqs (6) - (9)
# ============================================================================
def rhs(t: float, y: np.ndarray, u: np.ndarray,
        p: CSTRParams | None = None) -> np.ndarray:
    """Plant ODE rhs.

    Inputs:
      t  scalar time (min)
      y  (4,) state vector [C_A, T, T_c, h]
      u  (2,) input vector [Q_f, Q_c]   in L/min
      p  optional CSTRParams; defaults to Table 1 values.

    Returns dy/dt as a (4,) numpy array.
    """
    p = p or CSTRParams()
    C_A, T, T_c, h = float(y[0]), float(y[1]), float(y[2]), float(y[3])
    Q_f, Q_c = float(u[0]), float(u[1])

    # Floor h to avoid sqrt(<=0) and division-by-zero during transients.
    h_safe = max(h, p.h_min)
    sqrt_h = math.sqrt(h_safe)
    Vol = p.A * h_safe              # = A * h  (units chosen so that
                                    #   Q (L/min) / Vol gives 1/min)

    # Reaction rate (Arrhenius, first-order in C_A)
    T_safe = max(T, 1e-3)
    k = p.k_0 * math.exp(-p.Ea_over_R / T_safe)
    rA = k * C_A                    # mol/(L*min)

    # Mass balance, Eq (6)
    dC_A_dt = -rA + (Q_f * p.C_Af - p.C_V * sqrt_h * C_A) / Vol

    # Energy balance (reactor), Eq (7)
    Qrxn = rA * (-p.delta_H) / p.rho_Cp           # K/min
    Qconv = (Q_f * p.T_f - p.C_V * sqrt_h * T) / Vol
    Qjacket = p.UA_c * (T_c - T) / (p.rho_Cp * Vol)
    dT_dt = Qrxn + Qconv + Qjacket

    # Energy balance (coolant), Eq (8)
    dT_c_dt = (Q_c * (p.T_cf - T_c) / p.V_c
               + p.UA_c * (T - T_c) / (p.rho_c_Cpc * p.V_c))

    # Liquid level, Eq (9)
    dh_dt = (Q_f - p.C_V * sqrt_h) / p.A

    return np.array([dC_A_dt, dT_dt, dT_c_dt, dh_dt])


# ============================================================================
# One-step integrator  -  zero-order hold on u over dt
# ============================================================================
def step(y: np.ndarray, u: np.ndarray, dt: float,
         p: CSTRParams | None = None,
         rtol: float = 1e-8, atol: float = 1e-9) -> np.ndarray:
    """Advance the state by `dt` minutes under constant inputs u.

    Uses RK45 (scipy default) with tight tolerances.
    Returns the new state as a (4,) numpy array.
    """
    p = p or CSTRParams()
    sol = solve_ivp(rhs, t_span=(0.0, dt), y0=y, args=(u, p),
                     method="RK45", rtol=rtol, atol=atol,
                     max_step=dt / 4.0)
    y_next = sol.y[:, -1].copy()
    # Physical floors / safety
    y_next[0] = max(y_next[0], 0.0)        # C_A >= 0
    y_next[3] = max(y_next[3], p.h_min)    # h   >= h_min
    y_next[1] = max(y_next[1], 1.0)        # T   sane
    y_next[2] = max(y_next[2], 1.0)        # T_c sane
    return y_next


# ============================================================================
# Convenience: roll out an episode under a control trajectory
# ============================================================================
def simulate(y0: np.ndarray, u_traj: np.ndarray, dt: float = 1.0,
              p: CSTRParams | None = None) -> np.ndarray:
    """Simulate the plant forward.

    Inputs:
      y0      (4,) initial state
      u_traj  (N, 2) inputs at each of N steps (zero-order hold)
      dt      sample interval, min (paper uses 1 min)
    Returns:
      y_traj  (N+1, 4) state trajectory (including y0 at index 0)
    """
    p = p or CSTRParams()
    N = u_traj.shape[0]
    y_traj = np.zeros((N + 1, 4))
    y_traj[0] = y0
    y = y0.copy()
    for k in range(N):
        y = step(y, u_traj[k], dt, p)
        y_traj[k + 1] = y
    return y_traj


# ============================================================================
# Self-test: reproduce Appendix Fig A.1 (open-loop, Q_f=120, Q_c=15)
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Thosar 2025 CSTR plant - reproducing Fig A.1 (open-loop)")
    print("=" * 70)

    p = CSTRParams()
    print(f"\nParameters used:")
    print(f"  k_0       = {p.k_0:.2e} 1/min")
    print(f"  E_a/R     = {p.Ea_over_R}   K")
    print(f"  C_V       = {p.C_V}    L/(min*sqrt(m))   "
          f"[paper Table 1: 400, calibrated to y_SS]")
    print(f"  A         = {p.A}     m^2")
    print(f"  rho*C_P   = {p.rho_Cp}   J/(L*K)")
    print(f"  UA_c      = {p.UA_c}    J/(min*K)")
    print(f"  V_c       = {p.V_c}     L")
    print()

    # Initial conditions from Appendix
    y0 = np.array([0.0, 298.15, 298.15, 0.05])   # slight h>0 to avoid sqrt(0)
    u  = np.array([120.0, 15.0])
    T_end = 300.0   # min
    dt   = 1.0
    N    = int(T_end / dt)

    print(f"Initial: y0 = {y0}")
    print(f"Inputs:  Q_f = {u[0]} L/min, Q_c = {u[1]} L/min")
    print(f"Horizon: {T_end} min, dt = {dt} min ({N} steps)")
    print()

    u_traj = np.tile(u, (N, 1))
    y_traj = simulate(y0, u_traj, dt=dt, p=p)
    y_ss = y_traj[-1]

    # Paper Appendix says: y_SS = [0.0025, 416.1167, 351.5503, 9]
    paper_yss = np.array([0.0025, 416.1167, 351.5503, 9.0])
    err = np.abs(y_ss - paper_yss)
    rel = err / np.abs(paper_yss)

    print("Steady-state comparison:")
    print(f"  {'state':<8}{'ours':>14}{'paper':>14}{'abs err':>14}{'rel err':>10}")
    print(f"  {'-'*60}")
    for name, ours, ref, e, r in zip(
            ["C_A", "T", "T_c", "h"], y_ss, paper_yss, err, rel):
        print(f"  {name:<8}{ours:>14.6f}{ref:>14.6f}{e:>14.6f}{r:>10.2%}")
    print()

    # Sanity check
    max_rel = float(np.max(rel))
    if max_rel < 0.05:
        print(f"PASS - all states within 5% of paper y_SS "
              f"(max relative error {max_rel:.2%})")
    else:
        print(f"WARN - max relative error {max_rel:.2%} exceeds 5% "
              "- parameter calibration may be off")

    # Print intermediate trajectory for sanity
    print()
    print("Trajectory snapshots:")
    print(f"  {'t (min)':<10}{'C_A':>12}{'T (K)':>12}{'T_c (K)':>12}{'h (m)':>10}")
    for k in [0, 10, 30, 60, 100, 200, 300]:
        if k < y_traj.shape[0]:
            row = y_traj[k]
            print(f"  {k:<10}{row[0]:>12.5f}{row[1]:>12.3f}"
                  f"{row[2]:>12.3f}{row[3]:>10.3f}")
