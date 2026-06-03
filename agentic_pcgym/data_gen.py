"""Training data generator for the PC-Gym case studies.

For each case study, samples N random (initial-state, setpoint) episodes,
queries the NMPC oracle to get the first control action u_NMPC, and saves
the supervised pairs.

These pairs are used as the "conditioning" inputs to the PINN training
loop (the PINN learns to map (x_0, sp, u_IC) -> trajectory).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .nmpc_crystallization import (
    CrystallizationNMPC, CrystOperatingPoint, CrystBounds)
from .nmpc_fourtank import (
    FourTankNMPC, FourTankOperatingPoint, FourTankBounds)


# ============================================================================
# Crystallization episodes
# ============================================================================
def sample_crystallization_episodes(N: int, seed: int = 0,
                                      query_nmpc: bool = True,
                                      verbose: bool = True) -> dict:
    """Sample N random (mu0, mu1, mu2, mu3, c, cv_sp, ln_sp, Tc) episodes.

    Initial conditions are sampled to maintain L_n(0) ~ 5..25 and CV(0) ~ 0.5..3
    (consistent with PC-Gym Fig 6 operating range).

    If query_nmpc=True, queries the NMPC oracle for each and stores u_NMPC.
    """
    rng = np.random.default_rng(seed)
    op = CrystOperatingPoint()
    b = CrystBounds()
    # Sample log-uniform mu_0 and derive consistent moments
    mu0 = np.exp(rng.uniform(np.log(0.1), np.log(10.0), N)).astype(np.float32)
    # L_n in [5, 25]
    Ln = rng.uniform(5.0, 25.0, N).astype(np.float32)
    mu1 = mu0 * Ln
    # CV in [0.5, 3.0]: CV^2 = mu_2*mu_0/mu_1^2 - 1
    CV = rng.uniform(0.5, 3.0, N).astype(np.float32)
    mu2 = (CV * CV + 1) * mu1 * mu1 / mu0
    # mu_3 ~ mu_0 * L_n^3 (volume-equivalent)
    mu3 = mu0 * Ln ** 3
    # Concentration: random in [0.1, 0.5]
    c = rng.uniform(0.1, 0.5, N).astype(np.float32)
    # Setpoints
    cv_sp = rng.uniform(0.5, 2.5, N).astype(np.float32)
    ln_sp = rng.uniform(5.0, 25.0, N).astype(np.float32)
    # T_c IC
    Tc = rng.uniform(b.T_c_min, b.T_c_max, N).astype(np.float32)

    out = {"mu0_all": torch.from_numpy(mu0),
            "mu1_all": torch.from_numpy(mu1),
            "mu2_all": torch.from_numpy(mu2),
            "mu3_all": torch.from_numpy(mu3),
            "c_all":   torch.from_numpy(c),
            "cv_sp_all": torch.from_numpy(cv_sp),
            "ln_sp_all": torch.from_numpy(ln_sp),
            "Tc_all":  torch.from_numpy(Tc),
            "u_nmpc":  None}

    if query_nmpc:
        if verbose:
            print(f"Querying crystallization NMPC for {N} episodes...")
        oracle = CrystallizationNMPC()
        u_arr = np.zeros(N, dtype=np.float32)
        n_fail = 0
        t0 = time.time()
        for i in range(N):
            x_state = np.array([mu0[i], mu1[i], mu2[i], mu3[i], c[i]])
            u = oracle.query(x_state, sp_CV=cv_sp[i], sp_Ln=ln_sp[i])
            if u is None:
                n_fail += 1
                u_arr[i] = (b.T_c_min + b.T_c_max) / 2
            else:
                u_arr[i] = u
            if verbose and (i+1) % max(1, N//10) == 0:
                rate = (i+1) / (time.time() - t0)
                print(f"  [{i+1:>5}/{N}]  fail={n_fail}  ({rate:.1f}/s)")
        out["u_nmpc"] = torch.from_numpy(u_arr)
        if verbose:
            print(f"  Done. Failures: {n_fail}/{N}")
    return out


# ============================================================================
# Four-tank episodes
# ============================================================================
def sample_fourtank_episodes(N: int, seed: int = 0,
                                query_nmpc: bool = True,
                                verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    b = FourTankBounds()
    # Heights in [0, 1.0] m (within bounds; setpoints might be higher)
    h1 = rng.uniform(0.0, 1.0, N).astype(np.float32)
    h2 = rng.uniform(0.0, 1.0, N).astype(np.float32)
    h3 = rng.uniform(0.0, 1.0, N).astype(np.float32)
    h4 = rng.uniform(0.0, 1.0, N).astype(np.float32)
    # Setpoints (reachable: 0.1..0.8)
    h1_sp = rng.uniform(0.1, 0.8, N).astype(np.float32)
    h2_sp = rng.uniform(0.1, 0.6, N).astype(np.float32)
    # Initial voltages
    v1 = rng.uniform(b.v_min, b.v_max, N).astype(np.float32)
    v2 = rng.uniform(b.v_min, b.v_max, N).astype(np.float32)

    out = {"h1_all": torch.from_numpy(h1),
            "h2_all": torch.from_numpy(h2),
            "h3_all": torch.from_numpy(h3),
            "h4_all": torch.from_numpy(h4),
            "h1_sp_all": torch.from_numpy(h1_sp),
            "h2_sp_all": torch.from_numpy(h2_sp),
            "v1_all": torch.from_numpy(v1),
            "v2_all": torch.from_numpy(v2),
            "u_nmpc": None}

    if query_nmpc:
        if verbose:
            print(f"Querying four-tank NMPC for {N} episodes...")
        oracle = FourTankNMPC()
        u_arr = np.zeros((N, 2), dtype=np.float32)
        n_fail = 0
        t0 = time.time()
        for i in range(N):
            x_state = np.array([h1[i], h2[i], h3[i], h4[i]])
            u = oracle.query(x_state, sp_h1=h1_sp[i], sp_h2=h2_sp[i],
                              u_warm=(float(v1[i]), float(v2[i])))
            if u is None:
                n_fail += 1
                u_arr[i] = [(b.v_min + b.v_max) / 2] * 2
            else:
                u_arr[i] = u
            if verbose and (i+1) % max(1, N//10) == 0:
                rate = (i+1) / (time.time() - t0)
                print(f"  [{i+1:>5}/{N}]  fail={n_fail}  ({rate:.1f}/s)")
        out["u_nmpc"] = torch.from_numpy(u_arr)
        if verbose:
            print(f"  Done. Failures: {n_fail}/{N}")
    return out


def save_episodes(episodes: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Convert tensors to numpy for torch.save compactness
    out = {k: (v.numpy() if isinstance(v, torch.Tensor) else v)
           for k, v in episodes.items() if v is not None}
    torch.save(out, path)


def load_episodes(path: str) -> dict:
    data = torch.load(path, weights_only=False)
    return {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
            for k, v in data.items()}


if __name__ == "__main__":
    print("=== Smoke test: data generators ===\n")
    print("Crystallization (N=10, no NMPC):")
    eps_c = sample_crystallization_episodes(N=10, seed=0, query_nmpc=False)
    print(f"  mu_0 range: [{eps_c['mu0_all'].min():.3f}, {eps_c['mu0_all'].max():.3f}]")
    print(f"  L_n(0) range: ", (eps_c['mu1_all'] / eps_c['mu0_all']).numpy())
    print(f"\nFour-tank (N=10, no NMPC):")
    eps_f = sample_fourtank_episodes(N=10, seed=0, query_nmpc=False)
    print(f"  h_1 range: [{eps_f['h1_all'].min():.3f}, {eps_f['h1_all'].max():.3f}]")
    print(f"  setpoints OK: h1_sp range [{eps_f['h1_sp_all'].min():.3f}, {eps_f['h1_sp_all'].max():.3f}]")
