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

# APRBS hold length range (paper §A: "Sample durations were also randomly
# assigned within the range of 200-250 min to ensure that the system reached
# a steady state after each input change")
HOLD_MIN_STEPS = 200
HOLD_MAX_STEPS = 250


# ============================================================================
# APRBS input generator
# ============================================================================
def gen_aprbs(N: int, u_lo: np.ndarray, u_hi: np.ndarray,
              hold_min: int = HOLD_MIN_STEPS, hold_max: int = HOLD_MAX_STEPS,
              seed: int = 0) -> np.ndarray:
    """APRBS where EACH input channel has its OWN independent step schedule.

    Paper §A says "Sample durations were also randomly assigned within the
    range of 200-250 min" (plural). With independent Q_f and Q_c schedules,
    in 5000 min you get ~22 Q_f levels x ~22 Q_c levels ~ 484 distinct
    (Q_f, Q_c) pairs in the training trajectory - dense enough to cover
    the corners required by Test Case 1.

    With a SHARED schedule (both channels stepping together) you only get
    ~22 distinct (Q_f, Q_c) pairs in the full 5000 min and ~9 in the
    first 2000 min, which is too sparse to cover (100, 20) and (140, 10).

    Returns u of shape (N, n_u).
    """
    rng = np.random.default_rng(seed)
    n_u = u_lo.shape[0]
    u = np.zeros((N, n_u), dtype=float)
    # Independent APRBS per channel
    for c in range(n_u):
        i = 0
        while i < N:
            hold = int(rng.integers(hold_min, hold_max + 1))
            amp  = float(rng.uniform(u_lo[c], u_hi[c]))
            end  = min(i + hold, N)
            u[i:end, c] = amp
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


def gen_train_val_split(N_total: int = 5000, N_train: int = 2000,
                          seed: int = 0,
                          p: CSTRParams | None = None) -> tuple[dict, dict]:
    """Paper §A protocol (NOT the canonical baseline for this project - see
    `gen_grid_train_val_split` instead).

    Single 5000-min APRBS trajectory with Q_f in [100,140], Q_c in [10,20],
    200-250 min holds. First N_train min train; remaining val. Paper claims
    NARX MAE = 0.001508 with this; our reproduction shows MAE ~ 0.015 (10x
    worse) because random APRBS with ~22 amplitudes in 5000 min essentially
    never lands on the test corners (100,20) and (140,10). Keeping this for
    diagnostic / paper-protocol-audit purposes only.

    Returns (train_dict, val_dict).
    """
    u_lo = np.array([TRAIN_QF_RANGE[0], TRAIN_QC_RANGE[0]])
    u_hi = np.array([TRAIN_QF_RANGE[1], TRAIN_QC_RANGE[1]])
    u_full = gen_aprbs(N_total, u_lo, u_hi, seed=seed)
    y_full = _rollout(u_full, p=p)
    u_tr, u_va = u_full[:N_train],     u_full[N_train:]
    y_tr = y_full[:N_train + 1]
    y_va = y_full[N_train:]
    return ({"u": u_tr, "y": y_tr, "dt": DT_DEFAULT,
              "meta": f"train  N={N_train} of {N_total} seed={seed} (APRBS)"},
             {"u": u_va, "y": y_va, "dt": DT_DEFAULT,
              "meta": f"val    N={N_total-N_train} of {N_total} seed={seed} (APRBS)"})


# ============================================================================
# Canonical training protocol for this project: DENSE GRID
# ============================================================================
def gen_grid_train_val_split(qf_levels: int = 10, qc_levels: int = 10,
                                hold_min: int = 60, hold_max: int = 80,
                                train_frac: float = 0.6, seed: int = 0,
                                exclude_corners: bool = True,
                                n_train_points: int | None = None,
                                p: CSTRParams | None = None
                                ) -> tuple[dict, dict]:
    """Dense-grid training data: `qf_levels` x `qc_levels` Cartesian product
    of (Q_f, Q_c) amplitudes spanning the operating envelope, each held for
    a random duration in [hold_min, hold_max] minutes. Order is randomly
    shuffled so the model sees diverse transients.

    By default `exclude_corners=True`: the grid uses the OPEN interval
    (102, 138) x (10.5, 19.5) so it never hits the exact Test 1 schedule
    amplitudes (100, 20) and (140, 10). This avoids leakage from training
    onto the test corners and gives an honest evaluation of the model's
    interpolation/extrapolation capability.

    Set `exclude_corners=False` to use [100, 140] x [10, 20] (closed
    interval), which DOES include the exact test schedule values - useful
    only for sanity checks; not the canonical protocol.

    Total trajectory: qf_levels * qc_levels * mean_hold ~ 100 amplitudes x
    70 min = 7000 min. Train = first `train_frac`, val = remainder.

    If `n_train_points` is given, the training trajectory is truncated to
    EXACTLY that many time steps (the validation set is unchanged). Used
    for paper Fig 5/6 / Table 4 ablation (200 / 500 / 1000 / 2000 points).

    Returns (train_dict, val_dict).
    """
    rng = np.random.default_rng(seed)
    if exclude_corners:
        # Pad inward from each edge so (100, 20) and (140, 10) are NOT
        # hit by training amplitudes. The padding (2.0 for Q_f, 0.5 for Q_c)
        # is ~5% of each range.
        qf_lo, qf_hi = TRAIN_QF_RANGE[0] + 2.0, TRAIN_QF_RANGE[1] - 2.0
        qc_lo, qc_hi = TRAIN_QC_RANGE[0] + 0.5, TRAIN_QC_RANGE[1] - 0.5
    else:
        qf_lo, qf_hi = TRAIN_QF_RANGE
        qc_lo, qc_hi = TRAIN_QC_RANGE
    qf_vals = np.linspace(qf_lo, qf_hi, qf_levels)
    qc_vals = np.linspace(qc_lo, qc_hi, qc_levels)
    amps = [(float(qf), float(qc)) for qf in qf_vals for qc in qc_vals]
    rng.shuffle(amps)
    holds = rng.integers(hold_min, hold_max + 1, size=len(amps))
    N = int(holds.sum())
    u = np.zeros((N, 2), dtype=float)
    t = 0
    for (qf, qc), h in zip(amps, holds):
        u[t:t + h, 0] = qf
        u[t:t + h, 1] = qc
        t += h
    y = _rollout(u, p=p)
    n_train = int(N * train_frac)
    # Optional truncation to a specific number of training points (Fig 5/6)
    if n_train_points is not None:
        n_train = min(n_train_points, N - 100)   # always leave >=100 val pts
    tr = {"u": u[:n_train], "y": y[:n_train + 1], "dt": DT_DEFAULT,
           "meta": f"grid train N={n_train} of {N} "
                     f"({qf_levels}x{qc_levels} amps, holds {hold_min}-{hold_max}min, seed={seed})"}
    va = {"u": u[n_train:], "y": y[n_train:], "dt": DT_DEFAULT,
           "meta": f"grid val N={N-n_train} of {N} (same protocol)"}
    return tr, va


# Back-compat shims so call-sites that already use the old names keep working.
def gen_training_set(N: int = 2000, seed: int = 0,
                      p: CSTRParams | None = None) -> dict:
    """Paper-faithful: train = first 2000 min of the single 5000-min run."""
    tr, _ = gen_train_val_split(N_total=5000, N_train=N, seed=seed, p=p)
    return tr


def gen_validation_set(N: int = 3000, seed: int = 0,
                        p: CSTRParams | None = None) -> dict:
    """Paper-faithful: val = remaining 3000 min of the same 5000-min run.

    NOTE: `seed` should match the training seed - validation IS the second
    portion of the SAME trajectory, not an independent draw.
    """
    _, va = gen_train_val_split(N_total=N + 2000, N_train=2000, seed=seed, p=p)
    return va


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
    """Test Case 1 (paper §Generating test cases): TWO perturbations within
    the training range; SS in between.

       0-100    Q_f=120,  Q_c=15   (SS)
     100-300    Q_f=100,  Q_c=20   (perturbation #1, 200 min)
     300-900    Q_f=120,  Q_c=15   (returned to SS for ~600 min)
     900-1100   Q_f=140,  Q_c=10   (perturbation #2, 200 min)
    1100-1400   Q_f=120,  Q_c=15   (returned to SS, 300 min)
    """
    schedule = [
        (   0,  100, 120.0, 15.0),
        ( 100,  300, 100.0, 20.0),
        ( 300,  900, 120.0, 15.0),
        ( 900, 1100, 140.0, 10.0),
        (1100, 1400, 120.0, 15.0),
    ]
    out = _schedule_to_traj(schedule, p=p)
    out["meta"] = "test1_within_range (paper exact)"
    return out


def gen_test2_set(p: CSTRParams | None = None) -> dict:
    """Test Case 2 (paper §Generating test cases): same shape as Test 1 but
    each perturbation has BOTH Q_f and Q_c outside training range.

       0-100    Q_f=120,  Q_c=15   (SS)
     100-300    Q_f= 90,  Q_c=25   (perturbation #1 - both outside)
     300-900    Q_f=120,  Q_c=15   (back to SS)
     900-1100   Q_f=150,  Q_c= 5   (perturbation #2 - both outside)
    1100-1400   Q_f=120,  Q_c=15   (back to SS)
    """
    schedule = [
        (   0,  100, 120.0, 15.0),
        ( 100,  300,  90.0, 25.0),
        ( 300,  900, 120.0, 15.0),
        ( 900, 1100, 150.0,  5.0),
        (1100, 1400, 120.0, 15.0),
    ]
    out = _schedule_to_traj(schedule, p=p)
    out["meta"] = "test2_extrapolation (paper exact)"
    return out


# ============================================================================
# Noise wrapper - SNR-based Gaussian noise per channel  (Table 6)
# ============================================================================
def add_noise(y: np.ndarray, snr_per_channel: list[float],
               seed: int = 0) -> np.ndarray:
    """Add Gaussian measurement noise with the per-channel SNR.

    Paper §4.6 quotes "SNR represents the ratio of a signal's power to the
    corresponding noise amplitude" with SNR=35 (for C_A) and SNR=100 (for
    T, T_c, h). We use the **amplitude-ratio** convention:

        SNR  =  signal_std / noise_std

    so

        noise_std = signal_std / SNR

    This gives ~3% noise on C_A at SNR=35 and ~1% on T,T_c,h at SNR=100 -
    consistent with the visible-but-moderate noise in paper Figs 7-8 and
    with paper Table 6 PI-NARX MAEs (~0.009 at SNR=35). Earlier versions
    used the signal-processing power-ratio formula (signal_std / sqrt(SNR))
    which gave ~7x more noise on C_A and made our reproduction 5-7x worse
    than paper PI-NARX. See `git log -p data_gen.py` for the migration.

    Use SNR > 1e6 to effectively disable noise on a channel.
    """
    rng = np.random.default_rng(seed)
    y_noisy = y.copy().astype(np.float64)
    for c, snr in enumerate(snr_per_channel):
        sig = float(np.std(y[:, c]))
        if sig == 0.0 or not math.isfinite(snr) or snr <= 0.0 or snr > 1e6:
            continue
        sigma_noise = sig / snr
        y_noisy[:, c] = y[:, c] + rng.normal(0.0, sigma_noise, size=y.shape[0])
    return y_noisy.astype(y.dtype)


# Per-channel SNR vectors.
# Label refers to the C_A SNR ratio (paper Table 6 convention).
# T, T_c, h stay at SNR=100 in the paper-faithful profiles.
# Stress profiles (*_t3x) use 3x the per-amplitude noise on T and T_c
# (effective SNR ~ 33.33) - used to test residual_l2 sensitivity.
_T_SNR_3X = 100.0 / 3.0

NOISE_PROFILES = {
    # Paper-faithful: T,T_c,h all at SNR=100
    "snr35":  [35.0,  100.0, 100.0, 100.0],   # paper Table 6 case 1
    "snr75":  [75.0,  100.0, 100.0, 100.0],   # new mid-noise sweep
    "snr100": [100.0, 100.0, 100.0, 100.0],   # paper Table 6 case 2
    "snr125": [125.0, 100.0, 100.0, 100.0],   # new low-noise sweep
    "snr250": [250.0, 100.0, 100.0, 100.0],   # very-light C_A noise
    # Stress profiles: 3x the T,T_c noise (~3% of signal each)
    "snr35_t3x":  [35.0,  _T_SNR_3X, _T_SNR_3X, 100.0],
    "snr100_t3x": [100.0, _T_SNR_3X, _T_SNR_3X, 100.0],
}

# Back-compat aliases for callers still importing the old constants.
SNR_T6_NOISY_LO  = NOISE_PROFILES["snr35"]
SNR_T6_NOISY_HI  = NOISE_PROFILES["snr100"]
SNR_T6_NOISY_VHI = NOISE_PROFILES["snr250"]


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
