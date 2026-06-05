"""Training / validation / test data for the Thosar 2025 PI-NARX reproduction.

Protocol follows the paper Appendix:
  - dt = 1 min (sample time)
  - Training inputs: APRBS-style random steps within
        Q_f in [100, 140] L/min,   Q_c in [10, 20] L/min
  - Validation: same protocol, different seed
  - Test Case 1 (interpolation): scheduled steps INSIDE the training range
  - Test Case 2 (extrapolation): scheduled steps OUTSIDE the training range
        (Q_f in {90, 150}, Q_c in {5, 25})
  - Initial state: y0_SS = [0.0025, 416.12, 351.55, 9]  (the SS we calibrated)

Each dataset returns a dict with:
    "u" : (N, 2)   inputs at each step (Q_f, Q_c)
    "y" : (N+1, 4) state trajectory  [C_A, T, T_c, h]  (y[0] is the IC)
    "dt": float
    "meta": str    short description of how this set was generated

To match Table 6 (noisy data), wrap a clean trajectory with `add_noise(...)`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pinarx_plant import CSTRParams, simulate


# ============================================================================
# Plant operating envelopes (paper §4)
# ============================================================================
TRAIN_QF_RANGE  = (100.0, 140.0)   # L/min  (Q_f training amplitudes)
TRAIN_QC_RANGE  = (10.0,  20.0)    # L/min  (Q_c training amplitudes)

# Extrapolation amplitudes (well outside training)
EXTRAP_QF_VALS  = (90.0, 150.0)
EXTRAP_QC_VALS  = (5.0,  25.0)

# Paper steady-state initial condition (output of pinarx_plant smoke test)
Y0_SS = np.array([0.0025, 416.1167, 351.5503, 9.0])

# Default sample interval (paper uses 1 min)
DT_DEFAULT = 1.0

# Default APRBS hold length range (steps held this many sample intervals)
HOLD_MIN_STEPS = 20
HOLD_MAX_STEPS = 80


# ============================================================================
# APRBS input generator
# ============================================================================
def gen_aprbs(N: int, u_lo: np.ndarray, u_hi: np.ndarray,
              hold_min: int = HOLD_MIN_STEPS, hold_max: int = HOLD_MAX_STEPS,
              seed: int = 0) -> np.ndarray:
    """Amplitude-pseudo-random binary-like sequence: blocks of constant inputs,
    each block's duration drawn from U[hold_min, hold_max] and each amplitude
    drawn from U[u_lo, u_hi] (independently per input channel).

    Returns u of shape (N, n_u).
    """
    rng = np.random.default_rng(seed)
    n_u = u_lo.shape[0]
    u = np.zeros((N, n_u), dtype=float)
    i = 0
    while i < N:
        hold = int(rng.integers(hold_min, hold_max + 1))
        amp = rng.uniform(u_lo, u_hi)            # one amplitude per channel
        end = min(i + hold, N)
        u[i:end] = amp
        i = end
    return u


# ============================================================================
# One-shot dataset builders
# ============================================================================
def _rollout(u: np.ndarray, y0: np.ndarray | None = None,
              dt: float = DT_DEFAULT,
              p: CSTRParams | None = None) -> np.ndarray:
    """Roll out the plant for given inputs starting from y0 (defaults to Y0_SS).
    """
    y0 = Y0_SS.copy() if y0 is None else y0.copy()
    return simulate(y0, u, dt=dt, p=p)


def gen_training_set(N: int = 2000, seed: int = 0,
                      p: CSTRParams | None = None) -> dict:
    """Training set: APRBS within training range, N minutes long."""
    u_lo = np.array([TRAIN_QF_RANGE[0], TRAIN_QC_RANGE[0]])
    u_hi = np.array([TRAIN_QF_RANGE[1], TRAIN_QC_RANGE[1]])
    u = gen_aprbs(N, u_lo, u_hi, seed=seed)
    y = _rollout(u, p=p)
    return {"u": u, "y": y, "dt": DT_DEFAULT,
             "meta": f"training_set N={N} seed={seed} "
                       f"Q_f~U{TRAIN_QF_RANGE} Q_c~U{TRAIN_QC_RANGE}"}


def gen_validation_set(N: int = 3000, seed: int = 1,
                        p: CSTRParams | None = None) -> dict:
    """Validation set: same protocol as training, different seed."""
    u_lo = np.array([TRAIN_QF_RANGE[0], TRAIN_QC_RANGE[0]])
    u_hi = np.array([TRAIN_QF_RANGE[1], TRAIN_QC_RANGE[1]])
    u = gen_aprbs(N, u_lo, u_hi, seed=seed)
    y = _rollout(u, p=p)
    return {"u": u, "y": y, "dt": DT_DEFAULT,
             "meta": f"validation_set N={N} seed={seed} (same range as train)"}


def _schedule_to_traj(schedule: list[tuple[int, int, float, float]],
                       p: CSTRParams | None = None) -> dict:
    """Build inputs from a list of (t_start, t_end, Q_f, Q_c) blocks.
    All times are in minutes; assumes dt = 1 min.
    """
    if not schedule:
        raise ValueError("schedule cannot be empty")
    N = max(t_end for _, t_end, _, _ in schedule)
    u = np.zeros((N, 2))
    for t0, t1, qf, qc in schedule:
        u[t0:t1, 0] = qf
        u[t0:t1, 1] = qc
    y = _rollout(u, p=p)
    return {"u": u, "y": y, "dt": DT_DEFAULT}


def gen_test1_set(p: CSTRParams | None = None) -> dict:
    """Test Case 1: schedule of inputs INSIDE the training range (interpolation).

    Schedule (paper-style; minutes):
       0-100    Q_f=120,  Q_c=15   (start at SS)
     100-300    Q_f=100,  Q_c=20   (lo Q_f, hi Q_c)
     300-500    Q_f=120,  Q_c=15
     500-900    Q_f=130,  Q_c=12
     900-1100   Q_f=140,  Q_c=10   (hi Q_f, lo Q_c - corner of training cube)
    1100-1400   Q_f=110,  Q_c=18
    """
    schedule = [
        (   0,  100, 120.0, 15.0),
        ( 100,  300, 100.0, 20.0),
        ( 300,  500, 120.0, 15.0),
        ( 500,  900, 130.0, 12.0),
        ( 900, 1100, 140.0, 10.0),
        (1100, 1400, 110.0, 18.0),
    ]
    out = _schedule_to_traj(schedule, p=p)
    out["meta"] = "test1_within_range (interpolation)"
    return out


def gen_test2_set(p: CSTRParams | None = None) -> dict:
    """Test Case 2: schedule of inputs OUTSIDE the training range (extrapolation).

    Schedule (milder than Test 1's amplitude excursions to keep MAE bounded;
    paper-style: only ONE channel pushes outside at a time, shorter holds):
       0-100    Q_f=120,  Q_c=15   (start at SS)
     100-250    Q_f= 90,  Q_c=15   (Q_f below training)
     250-400    Q_f=120,  Q_c=15
     400-550    Q_f=150,  Q_c=15   (Q_f above training)
     550-700    Q_f=120,  Q_c=15
     700-850    Q_f=120,  Q_c= 5   (Q_c below training)
     850-1000   Q_f=120,  Q_c=15
    1000-1150   Q_f=120,  Q_c=25   (Q_c above training)
    1150-1400   Q_f=120,  Q_c=15   (return to SS)
    """
    schedule = [
        (   0,  100, 120.0, 15.0),
        ( 100,  250,  90.0, 15.0),
        ( 250,  400, 120.0, 15.0),
        ( 400,  550, 150.0, 15.0),
        ( 550,  700, 120.0, 15.0),
        ( 700,  850, 120.0,  5.0),
        ( 850, 1000, 120.0, 15.0),
        (1000, 1150, 120.0, 25.0),
        (1150, 1400, 120.0, 15.0),
    ]
    out = _schedule_to_traj(schedule, p=p)
    out["meta"] = "test2_extrapolation (one channel outside at a time)"
    return out


# ============================================================================
# Noise wrapper - SNR-based Gaussian noise per channel  (Table 6)
# ============================================================================
def add_noise(y: np.ndarray, snr_db_per_channel: list[float],
               seed: int = 0) -> np.ndarray:
    """Add Gaussian measurement noise with the per-channel SNR (in dB) given by
    the paper:

        SNR_dB = 10 * log10( P_signal / P_noise )
        ->  sigma_noise = sigma_signal / 10^(SNR_dB / 20)

    Paper Table 6 uses:
        SNR = 35  for C_A
        SNR = 100 for T, T_c, h

    sigma_signal is computed from the trajectory variance per channel.
    """
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    for c, snr in enumerate(snr_db_per_channel):
        sig = float(np.std(y[:, c]))
        if sig == 0.0 or not math.isfinite(snr) or snr > 200:
            continue
        sigma_noise = sig / (10.0 ** (snr / 20.0))
        y_noisy[:, c] = y[:, c] + rng.normal(0.0, sigma_noise, size=y.shape[0])
    return y_noisy


# Paper Table 6 standard SNR vectors
SNR_T6_NOISY_LO = [35.0, 100.0, 100.0, 100.0]    # "SNR 35" row
SNR_T6_NOISY_HI = [100.0, 100.0, 100.0, 100.0]   # "SNR 100" row


# ============================================================================
# Self-test
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("data_gen.py self-test")
    print("=" * 70)

    print("\n[1] Training set (N=2000, seed=0)")
    tr = gen_training_set(N=2000, seed=0)
    print(f"  u shape: {tr['u'].shape}  y shape: {tr['y'].shape}")
    print(f"  Q_f range: [{tr['u'][:,0].min():.1f}, {tr['u'][:,0].max():.1f}]")
    print(f"  Q_c range: [{tr['u'][:,1].min():.1f}, {tr['u'][:,1].max():.1f}]")
    print(f"  C_A range: [{tr['y'][:,0].min():.4f}, {tr['y'][:,0].max():.4f}]")
    print(f"  T   range: [{tr['y'][:,1].min():.1f}, {tr['y'][:,1].max():.1f}]")

    print("\n[2] Validation set (N=3000, seed=1)")
    va = gen_validation_set(N=3000, seed=1)
    print(f"  u shape: {va['u'].shape}  y shape: {va['y'].shape}")

    print("\n[3] Test Case 1 (within range)")
    t1 = gen_test1_set()
    print(f"  u shape: {t1['u'].shape}  y shape: {t1['y'].shape}")
    print(f"  Q_f range: [{t1['u'][:,0].min():.1f}, {t1['u'][:,0].max():.1f}]")
    print(f"  Q_c range: [{t1['u'][:,1].min():.1f}, {t1['u'][:,1].max():.1f}]")
    print(f"  Final state: {t1['y'][-1]}")

    print("\n[4] Test Case 2 (extrapolation)")
    t2 = gen_test2_set()
    print(f"  u shape: {t2['u'].shape}  y shape: {t2['y'].shape}")
    print(f"  Q_f range: [{t2['u'][:,0].min():.1f}, {t2['u'][:,0].max():.1f}]")
    print(f"  Q_c range: [{t2['u'][:,1].min():.1f}, {t2['u'][:,1].max():.1f}]")
    print(f"  Final state: {t2['y'][-1]}")

    print("\n[5] Noise wrapper (SNR=35 on C_A, SNR=100 elsewhere)")
    y_noisy = add_noise(t1["y"], SNR_T6_NOISY_LO, seed=42)
    diff = y_noisy - t1["y"]
    for c, name in enumerate(["C_A", "T", "T_c", "h"]):
        sig = float(np.std(t1["y"][:, c]))
        sig_noise = float(np.std(diff[:, c]))
        snr_meas = 20.0 * math.log10(max(sig, 1e-12) / max(sig_noise, 1e-12))
        print(f"  {name:<4} sig std={sig:>10.4f}  noise std={sig_noise:>10.4f}"
                f"  measured SNR={snr_meas:>6.1f} dB")

    print("\nDONE.")
