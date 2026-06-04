"""Closed-loop evaluator with Bloor's optimality-gap + MAD metrics.

Implements the evaluation protocol from Bloor et al. 2025 (Sec 2.5):

  Optimality gap (Eq 35):
    Delta(pi) = ( J(pi_oracle) - J(pi) ) / N_e
        where J(pi) = median normalised cumulative reward across reps
              N_e   = number of discrete setpoint errors in an episode

  Median Absolute Deviation (Eq 10):
    MAD(pi) = median( | sum_t gamma^t r_t - J_tilde(pi) | )

Reward function (Bloor Eq 13, used for both NMPC and any controller):
    r(x_t, u_t) = -((x_bar_{t+1} - x_bar_sp)^T Q (...) + (u_bar_t - u_bar_{t-1})^T R (...))
where the bar indicates normalised variables.

For Bloor's case studies Q = identity, R = 0.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .plants.crystallization import (
    CrystParams, step as cryst_step, CV_from_moments, Ln_from_moments,
    CrystScenario)
from .plants.fourtank import (
    FourTankParams, step as fourtank_step, FourTankScenario)
from .plants.cstr import (
    CSTRParams, step as cstr_step, CSTRScenario)
from .nmpc_crystallization import (
    CrystallizationNMPC, CrystOperatingPoint, CrystBounds)
from .nmpc_fourtank import (
    FourTankNMPC, FourTankOperatingPoint, FourTankBounds)
from .nmpc_cstr import (
    CSTRNMPC, CSTROperatingPoint, CSTRBounds)


# ============================================================================
# Reward function (Bloor Eq 13) - normalised reference
# ============================================================================
def normalised_reward_crystallization(CV: float, Ln: float,
                                        u: float, u_prev: float,
                                        cv_sp: float, ln_sp: float,
                                        b: CrystBounds | None = None,
                                        ) -> float:
    """Bloor Eq 13 with Q=identity over [CV, L_n], R=0. Q=I, R=0 per paper."""
    b = b or CrystBounds()
    # Normalise outputs
    cv_n   = (CV - b.CV_min) / (b.CV_max - b.CV_min)
    ln_n   = (Ln - b.Ln_min) / (b.Ln_max - b.Ln_min)
    cvsp_n = (cv_sp - b.CV_min) / (b.CV_max - b.CV_min)
    lnsp_n = (ln_sp - b.Ln_min) / (b.Ln_max - b.Ln_min)
    err = (cv_n - cvsp_n) ** 2 + (ln_n - lnsp_n) ** 2
    return -err   # Q=I, R=0


def normalised_reward_fourtank(h1: float, h2: float,
                                v1: float, v2: float, v1_prev: float, v2_prev: float,
                                h1_sp: float, h2_sp: float,
                                b: FourTankBounds | None = None,
                                ) -> float:
    """Bloor Eq 13 with Q=identity over [h_1, h_2], R=0."""
    b = b or FourTankBounds()
    h_range = b.h_max - b.h_min
    e1 = (h1 - h1_sp) / h_range
    e2 = (h2 - h2_sp) / h_range
    return -(e1 * e1 + e2 * e2)


# ============================================================================
# Closed-loop simulation (crystallization)
# ============================================================================
@dataclass
class CrystEpisodeResult:
    cumulative_reward: float
    CV_trajectory: list
    Ln_trajectory: list
    Tc_trajectory: list
    c_trajectory: list
    n_steps: int


def closed_loop_crystallization(
        controller_query: Callable,   # (mu0,mu1,mu2,mu3,c, cv_sp, ln_sp) -> T_c
        x0: np.ndarray,
        cv_sp: float, ln_sp: float,
        n_steps: int = None,
        scen: CrystScenario | None = None,
        params: CrystParams | None = None,
        ) -> CrystEpisodeResult:
    """Run one closed-loop episode (30 hr, dt=1 hr by default)."""
    scen = scen or CrystScenario()
    params = params or CrystParams()
    n_steps = n_steps or scen.n_steps
    x = x0.copy()
    cumulative_reward = 0.0
    cv_traj, ln_traj, tc_traj, c_traj = [], [], [], []
    u_prev = 30.0    # initial T_c guess
    for k in range(n_steps):
        # Get current outputs
        cv = CV_from_moments(x[0], x[1], x[2])
        ln = Ln_from_moments(x[0], x[1])
        # Query controller
        u = controller_query(x[0], x[1], x[2], x[3], x[4], cv_sp, ln_sp)
        if u is None:
            u = u_prev
        # Step plant
        x = cryst_step(x, u, scen.dt_hr, params)
        # Reward AFTER step (Bloor convention: r(x_{t+1}, u_t))
        cv_next = CV_from_moments(x[0], x[1], x[2])
        ln_next = Ln_from_moments(x[0], x[1])
        r = normalised_reward_crystallization(cv_next, ln_next, u, u_prev,
                                                 cv_sp, ln_sp)
        cumulative_reward += r
        cv_traj.append(float(cv_next))
        ln_traj.append(float(ln_next))
        tc_traj.append(float(u))
        c_traj.append(float(x[4]))
        u_prev = u
    return CrystEpisodeResult(
        cumulative_reward=cumulative_reward,
        CV_trajectory=cv_traj, Ln_trajectory=ln_traj,
        Tc_trajectory=tc_traj, c_trajectory=c_traj, n_steps=n_steps)


# ============================================================================
# Closed-loop simulation (four-tank)
# ============================================================================
@dataclass
class FourTankEpisodeResult:
    cumulative_reward: float
    h_trajectory: list      # list of (h1, h2, h3, h4) tuples
    v_trajectory: list      # list of (v1, v2) tuples
    n_steps: int


def closed_loop_fourtank(
        controller_query: Callable,    # (h1, h2, h3, h4, h1_sp, h2_sp, v1_prev, v2_prev) -> (v1, v2)
        x0: np.ndarray,
        h1_sp, h2_sp,                   # float OR array of length n_steps (Bloor-style step schedule)
        n_steps: int = None,
        scen: FourTankScenario | None = None,
        params: FourTankParams | None = None,
        ) -> FourTankEpisodeResult:
    """Closed-loop four-tank simulation.

    Setpoints can be either:
      - scalars (constant throughout the episode, original behaviour), OR
      - 1-D arrays of length n_steps (Bloor Fig 7-style step schedule with
        within-episode setpoint changes).
    """
    scen = scen or FourTankScenario()
    params = params or FourTankParams()
    n_steps = n_steps or scen.n_steps
    # Auto-broadcast scalar setpoints to per-step schedules for backward compat
    h1_sp_sched = (np.full(n_steps, float(h1_sp), dtype=float)
                    if np.isscalar(h1_sp) else np.asarray(h1_sp, dtype=float))
    h2_sp_sched = (np.full(n_steps, float(h2_sp), dtype=float)
                    if np.isscalar(h2_sp) else np.asarray(h2_sp, dtype=float))
    if len(h1_sp_sched) != n_steps or len(h2_sp_sched) != n_steps:
        raise ValueError(
            f"Setpoint schedule length mismatch: h1={len(h1_sp_sched)}, "
            f"h2={len(h2_sp_sched)}, expected {n_steps}")
    x = x0.copy()
    cumulative_reward = 0.0
    h_traj, v_traj = [], []
    v_prev = (5.0, 5.0)
    for k in range(n_steps):
        h1_sp_k = float(h1_sp_sched[k])
        h2_sp_k = float(h2_sp_sched[k])
        u = controller_query(x[0], x[1], x[2], x[3], h1_sp_k, h2_sp_k,
                              v_prev[0], v_prev[1])
        if u is None:
            u = v_prev
        u = np.asarray(u, dtype=float)
        x = fourtank_step(x, u, scen.dt_s, params)
        r = normalised_reward_fourtank(x[0], x[1], u[0], u[1],
                                          v_prev[0], v_prev[1],
                                          h1_sp_k, h2_sp_k)
        cumulative_reward += r
        h_traj.append((float(x[0]), float(x[1]), float(x[2]), float(x[3])))
        v_traj.append((float(u[0]), float(u[1])))
        v_prev = (float(u[0]), float(u[1]))
    return FourTankEpisodeResult(
        cumulative_reward=cumulative_reward,
        h_trajectory=h_traj, v_trajectory=v_traj, n_steps=n_steps)


def make_bloor_setpoint_schedule(
        rng: np.random.Generator, n_steps: int = 60,
        jitter: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Bloor Fig 7-style 3-phase step setpoint schedule for the four-tank.

    Phase 1 (steps 0..k1):     h1 ~ 0.4, h2 ~ 0.2  (initial / hold)
    Phase 2 (steps k1..k2):    h1 ~ 0.6, h2 ~ 0.3  (step up)
    Phase 3 (steps k2..end):   h1 ~ 0.1, h2 ~ 0.2  (step down)

    If jitter=True, each phase's setpoint and the transition timesteps are
    perturbed (±10% / ±2 steps) so that different reps see different but
    Bloor-shaped trajectories — gives MAD a non-trivial spread.

    Returns (h1_sp_sched, h2_sp_sched) each of shape (n_steps,).
    """
    if jitter:
        k1 = int(np.clip(rng.normal(n_steps / 3.0, 1.5), 5, n_steps // 2))
        k2 = int(np.clip(rng.normal(2 * n_steps / 3.0, 1.5), k1 + 5, n_steps - 3))
        h1_vals = [
            float(np.clip(rng.normal(0.4, 0.04), 0.30, 0.50)),
            float(np.clip(rng.normal(0.6, 0.04), 0.50, 0.70)),
            float(np.clip(rng.normal(0.1, 0.03), 0.05, 0.20)),
        ]
        h2_vals = [
            float(np.clip(rng.normal(0.2, 0.03), 0.15, 0.30)),
            float(np.clip(rng.normal(0.3, 0.03), 0.25, 0.40)),
            float(np.clip(rng.normal(0.2, 0.03), 0.15, 0.30)),
        ]
    else:
        k1, k2 = n_steps // 3, 2 * n_steps // 3
        h1_vals = [0.4, 0.6, 0.1]
        h2_vals = [0.2, 0.3, 0.2]

    h1_sched = np.zeros(n_steps, dtype=float)
    h2_sched = np.zeros(n_steps, dtype=float)
    h1_sched[:k1] = h1_vals[0]
    h1_sched[k1:k2] = h1_vals[1]
    h1_sched[k2:] = h1_vals[2]
    h2_sched[:k1] = h2_vals[0]
    h2_sched[k1:k2] = h2_vals[1]
    h2_sched[k2:] = h2_vals[2]
    return h1_sched, h2_sched


# ============================================================================
# Metric aggregators (Bloor Eq 35 + Eq 10)
# ============================================================================
def optimality_gap(rewards_pi, reward_oracle: float, n_setpoint_errors: int = 1,
                    ) -> float:
    """Bloor Eq 35: Delta(pi) = (J(pi*) - J(pi)) / N_e ."""
    J_pi = float(np.median(rewards_pi))
    return (reward_oracle - J_pi) / max(1, n_setpoint_errors)


def median_absolute_deviation(rewards_pi) -> float:
    """Bloor Eq 10."""
    arr = np.asarray(rewards_pi, dtype=float)
    median = np.median(arr)
    return float(np.median(np.abs(arr - median)))


# ============================================================================
# Full evaluation: run controller across n_reps episodes, compare to oracle
# ============================================================================
def evaluate_crystallization(
        controller_query: Callable,
        n_reps: int = 50,
        seed: int = 0,
        verbose: bool = True,
        ) -> dict:
    """Bloor's standard CV+L_n setpoint-tracking evaluation."""
    rng = np.random.default_rng(seed)
    op = CrystOperatingPoint()
    rewards_pi = []
    rewards_oracle = []
    oracle = CrystallizationNMPC()
    oracle_query = lambda *a, **kw: oracle.query(np.array(a[:5]),
                                                    sp_CV=a[5], sp_Ln=a[6])
    for rep in range(n_reps):
        # CRITICAL: reset oracle warm-start between episodes. Within an
        # episode, sequential make_step() calls auto-warm-start from the
        # previous solution (do-mpc default behavior). Between episodes,
        # the previous solution is stale, so we reset.
        oracle.reset()
        x0 = np.array([op.mu_0_0, op.mu_1_0, op.mu_2_0, op.mu_3_0, op.c_0])
        # Add small randomisation
        x0 = x0 * rng.uniform(0.9, 1.1, 5).astype(np.float64)
        x0[4] = float(rng.uniform(0.2, 0.4))
        # Setpoints matched to Bloor 2025 Fig 6 protocol: TIGHT range around
        # the operating point (CV ~ 1.0, Ln ~ 15 um). Previous version sampled
        # cv_sp in [0.5, 2.0] which is 4x the operating range — even the NMPC
        # oracle can't track such extreme changes in 30 hours, so both PINN
        # and oracle saturated to bad rewards (~-17000) and opt_gap was
        # uninformative. Bloor's RL gets opt_gap ~0.01 because they stay near
        # the operating point. We now do the same.
        cv_sp = float(rng.uniform(0.9, 1.2))    # tight around CV ~ 1.0
        ln_sp = float(rng.uniform(14.0, 16.0))  # tight around Ln ~ 15 um
        # Run controller under test
        r_pi = closed_loop_crystallization(controller_query, x0, cv_sp, ln_sp)
        # Run oracle on same scenario
        r_or = closed_loop_crystallization(oracle_query, x0, cv_sp, ln_sp)
        rewards_pi.append(r_pi.cumulative_reward)
        rewards_oracle.append(r_or.cumulative_reward)
        if verbose and (rep + 1) % max(1, n_reps // 5) == 0:
            print(f"  rep {rep+1}/{n_reps}: pi={r_pi.cumulative_reward:.3f}, "
                  f"oracle={r_or.cumulative_reward:.3f}")
    gap = optimality_gap(rewards_pi, float(np.median(rewards_oracle)),
                          n_setpoint_errors=1)
    mad = median_absolute_deviation(rewards_pi)
    return {"rewards_pi": rewards_pi, "rewards_oracle": rewards_oracle,
             "optimality_gap": gap, "MAD": mad,
             "median_reward_pi":      float(np.median(rewards_pi)),
             "median_reward_oracle":  float(np.median(rewards_oracle))}


def evaluate_fourtank(controller_query: Callable, n_reps: int = 50,
                       seed: int = 0, verbose: bool = True,
                       step_setpoints: bool = False) -> dict:
    """Evaluate a controller against the NMPC oracle on the four-tank.

    step_setpoints:
      False (default) -> each episode has ONE random (h1_sp, h2_sp) pair
                         held constant for all 60 steps. This is our
                         original protocol (broader operating range, harder
                         distribution).
      True            -> each episode uses a Bloor Fig 7-style 3-phase step
                         schedule (h1: 0.4 -> 0.6 -> 0.1; h2: 0.2 -> 0.3 ->
                         0.2) with jittered timing/values. Apples-to-apples
                         comparison with Bloor 2025 Table 5.

    n_setpoint_errors for the optimality_gap is automatically set to 2 when
    step_setpoints=True (h1 and h2 each have 2 transitions = 2 setpoint
    errors per episode in Bloor's accounting), else 1.
    """
    rng = np.random.default_rng(seed)
    op = FourTankOperatingPoint()
    rewards_pi, rewards_oracle = [], []
    oracle = FourTankNMPC()
    def oracle_query(h1, h2, h3, h4, h1_sp, h2_sp, v1_prev, v2_prev):
        return oracle.query(np.array([h1, h2, h3, h4]),
                              sp_h1=h1_sp, sp_h2=h2_sp,
                              u_warm=(v1_prev, v2_prev))
    scen = FourTankScenario()
    for rep in range(n_reps):
        x0 = np.array([op.h_1_0, op.h_2_0, op.h_3_0, op.h_4_0])
        x0 = x0 * rng.uniform(0.8, 1.2, 4)
        if step_setpoints:
            h1_sp, h2_sp = make_bloor_setpoint_schedule(
                rng, n_steps=scen.n_steps, jitter=True)
        else:
            h1_sp = float(rng.uniform(0.2, 0.7))
            h2_sp = float(rng.uniform(0.2, 0.5))
        r_pi = closed_loop_fourtank(controller_query, x0, h1_sp, h2_sp)
        r_or = closed_loop_fourtank(oracle_query,        x0, h1_sp, h2_sp)
        rewards_pi.append(r_pi.cumulative_reward)
        rewards_oracle.append(r_or.cumulative_reward)
        if verbose and (rep + 1) % max(1, n_reps // 5) == 0:
            print(f"  rep {rep+1}/{n_reps}: pi={r_pi.cumulative_reward:.3f}, "
                  f"oracle={r_or.cumulative_reward:.3f}")
    n_setpoint_errors = 2 if step_setpoints else 1
    return {"rewards_pi": rewards_pi, "rewards_oracle": rewards_oracle,
             "optimality_gap": optimality_gap(rewards_pi,
                                                float(np.median(rewards_oracle)),
                                                n_setpoint_errors=n_setpoint_errors),
             "MAD": median_absolute_deviation(rewards_pi),
             "median_reward_pi":     float(np.median(rewards_pi)),
             "median_reward_oracle": float(np.median(rewards_oracle)),
             "step_setpoints": step_setpoints,
             "n_setpoint_errors": n_setpoint_errors}


# ============================================================================
# CSTR reward + closed-loop + evaluator
# ============================================================================
def normalised_reward_cstr(C_A: float, T_c: float, T_c_prev: float,
                            CA_sp: float,
                            b: CSTRBounds | None = None) -> float:
    """Bloor Eq 13 with Q=identity over [C_A], R=0."""
    b = b or CSTRBounds()
    CA_range = b.C_A_max - b.C_A_min
    e_CA = (C_A - CA_sp) / CA_range
    return -(e_CA * e_CA)


@dataclass
class CSTREpisodeResult:
    cumulative_reward: float
    C_A_trajectory: list
    T_trajectory: list
    T_c_trajectory: list
    n_steps: int


def closed_loop_cstr(
        controller_query: Callable,    # (C_A, T, CA_sp, T_c_prev) -> T_c
        x0: np.ndarray,
        CA_sp: float,
        n_steps: int = None,
        scen: CSTRScenario | None = None,
        params: CSTRParams | None = None,
        ) -> CSTREpisodeResult:
    scen = scen or CSTRScenario()
    params = params or CSTRParams()
    n_steps = n_steps or scen.n_steps
    x = x0.copy()
    cumulative_reward = 0.0
    CA_traj, T_traj, Tc_traj = [], [], []
    T_c_prev = 300.0   # initial T_c guess
    for k in range(n_steps):
        u = controller_query(float(x[0]), float(x[1]),
                              float(CA_sp), float(T_c_prev))
        if u is None:
            u = T_c_prev
        T_c = float(u)
        x = cstr_step(x, T_c, scen.dt_min, params)
        r = normalised_reward_cstr(float(x[0]), T_c, T_c_prev, float(CA_sp))
        cumulative_reward += r
        CA_traj.append(float(x[0]))
        T_traj.append(float(x[1]))
        Tc_traj.append(T_c)
        T_c_prev = T_c
    return CSTREpisodeResult(
        cumulative_reward=cumulative_reward,
        C_A_trajectory=CA_traj,
        T_trajectory=T_traj,
        T_c_trajectory=Tc_traj,
        n_steps=n_steps)


def evaluate_cstr(controller_query: Callable, n_reps: int = 50,
                   seed: int = 0, verbose: bool = True) -> dict:
    """Bloor's standard C_A setpoint-tracking evaluation."""
    rng = np.random.default_rng(seed)
    op = CSTROperatingPoint()
    rewards_pi, rewards_oracle = [], []
    oracle = CSTRNMPC()
    def oracle_query(C_A, T, CA_sp, T_c_prev):
        return oracle.query(np.array([C_A, T]), sp_CA=CA_sp)
    for rep in range(n_reps):
        # Per-episode warm-start reset (critical, learned from crystallization)
        oracle.reset()
        # Initial state with small randomisation
        x0 = np.array([op.C_A_0, op.T_0])
        x0[0] *= rng.uniform(0.95, 1.05)
        x0[1] *= rng.uniform(0.97, 1.03)
        # Setpoint within Bloor Fig 3 envelope
        CA_sp = float(rng.uniform(0.82, 0.91))
        r_pi = closed_loop_cstr(controller_query, x0, CA_sp)
        r_or = closed_loop_cstr(oracle_query,        x0, CA_sp)
        rewards_pi.append(r_pi.cumulative_reward)
        rewards_oracle.append(r_or.cumulative_reward)
        if verbose and (rep + 1) % max(1, n_reps // 5) == 0:
            print(f"  rep {rep+1}/{n_reps}: pi={r_pi.cumulative_reward:.3f}, "
                  f"oracle={r_or.cumulative_reward:.3f}")
    return {"rewards_pi": rewards_pi, "rewards_oracle": rewards_oracle,
             "optimality_gap": optimality_gap(rewards_pi,
                                                float(np.median(rewards_oracle)),
                                                n_setpoint_errors=1),
             "MAD": median_absolute_deviation(rewards_pi),
             "median_reward_pi":     float(np.median(rewards_pi)),
             "median_reward_oracle": float(np.median(rewards_oracle))}


if __name__ == "__main__":
    # Self-test: compare oracle-vs-oracle (gap should be ~0)
    print("=== Crystallization: oracle vs oracle (sanity check, gap ~0) ===")
    oracle = CrystallizationNMPC()
    q = lambda *a: oracle.query(np.array(a[:5]), sp_CV=a[5], sp_Ln=a[6])
    r = evaluate_crystallization(q, n_reps=3, verbose=True)
    print(f"  Optimality gap: {r['optimality_gap']:.6f}  (expect ~0)")
    print(f"  MAD: {r['MAD']:.6f}")
    print()
    print("=== Four-tank: oracle vs oracle (sanity check, gap ~0) ===")
    fto = FourTankNMPC()
    def fq(h1, h2, h3, h4, h1_sp, h2_sp, v1_p, v2_p):
        return fto.query(np.array([h1, h2, h3, h4]),
                          sp_h1=h1_sp, sp_h2=h2_sp, u_warm=(v1_p, v2_p))
    r2 = evaluate_fourtank(fq, n_reps=3, verbose=True)
    print(f"  Optimality gap: {r2['optimality_gap']:.6f}  (expect ~0)")
    print(f"  MAD: {r2['MAD']:.6f}")
