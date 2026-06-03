"""paper_figures.py - Reproduce Kardamaki 2026 paper figures + tables with
our improved (LLM-AutoOpt + DPC) PINN-MPC.

Outputs (saved under --output dir):
  fig04_loss_curves.png         (Kardamaki Fig 4)
  fig05_multistep_tracking.png  (Kardamaki Fig 5)
  fig06_multistep_with_noise.png (Kardamaki Fig 6)
  fig07_model_mismatch.png      (Kardamaki Fig 7)
  fig11_single_scenario.png     (Kardamaki Fig 11)
  table03_loss_decomposition.csv (Kardamaki Table 3)
  table04_performance_metrics.csv (Kardamaki Table 4)
  table10_runtime.csv           (Kardamaki Table 10)
  summary.json                  (all metrics in one place)

Usage:
  # After STEP 2 has produced results/llm_phaseA/llm_trials.json
  python paper_figures.py \
      --best-cfg results/llm_phaseA/llm_trials.json \
      --output results/paper_figures/

  # To skip the costly NMPC comparison (only needed for Fig 11):
  python paper_figures.py --skip-nmpc

  # To run only specific outputs:
  python paper_figures.py --only fig11,table10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize

from agentic_pinn_mpc.pinn_siso import (DEVICE, PINNHparams, PINN_Controller,
                                          train_pinn_siso)
from agentic_pinn_mpc.evaluate import evaluate_model
from agentic_pinn_mpc.bench import load_training_data
from agentic_pinn_mpc.tuners import KARDAMAKI_BEST
from agentic_pinn_mpc.rl_refine import refine_with_dpc, DPCRefineCfg


# ============================================================
# Constants
# ============================================================
K_VALVE = 0.7
A_DEFAULT = 1.0
DT_PLANT = 0.01      # plant integration step (s)
TS_CTRL = 1.0        # controller sampling time (s)
U_MIN, U_MAX = 0.0, 1.0
X_MIN, X_MAX = 0.0, 4.0
DU_MAX = 0.2          # Kardamaki Sec 4.1.1 - rate constraint per controller step

KARDAMAKI_TABLE4 = {
    "tracking_mean_offset_m":   0.0161,
    "tracking_max_offset_m":    0.0322,
    "disturbance_mean_offset_m":0.0129,
    "disturbance_max_offset_m": 0.0453,
}
KARDAMAKI_TABLE10 = {
    "pinn_mean_ms":  0.061,
    "pinn_std_ms":   0.009,
    "pinn_max_ms":   0.176,
    "pinn_max_std":  0.110,
    "pinn_total_ms": 3.035,
    "pinn_total_std":0.430,
    "nmpc_mean_ms":  25.390,
    "nmpc_std_ms":   4.624,
    "nmpc_max_ms":   54.608,
    "nmpc_max_std":  30.419,
    "nmpc_total_ms": 1269.521,
    "nmpc_total_std":231.224,
}


# ============================================================
# Plant simulation (Euler integration, Kardamaki dynamics)
# ============================================================
def plant_step(x: float, u: float, d: float,
                K: float = K_VALVE, A: float = A_DEFAULT,
                dt: float = DT_PLANT, n_inner: int = 100) -> float:
    """Integrate dx/dt = (u + d - K*sqrt(x))/A for one controller step Ts."""
    for _ in range(n_inner):
        sqrt_x = math.sqrt(max(x, 0.0))
        dx = (u + d - K * sqrt_x) / A
        x = max(x + dx * dt, 0.0)
    return x


# ============================================================
# NMPC oracle (scipy-based, mirrors Kardamaki's setup)
# ============================================================
class NMPCOracle:
    """SISO NMPC: minimise tracking error + control effort over N steps.

    Used only for Fig 11 PINN-vs-NMPC comparison and Table 10 runtime
    statistics. Single-shooting SQP via scipy.
    """
    def __init__(self, N: int = 10, R: float = 0.01,
                 K: float = K_VALVE, A: float = A_DEFAULT,
                 Ts: float = TS_CTRL, dt_inner: float = 0.1):
        self.N = N
        self.R = R
        self.K = K
        self.A = A
        self.Ts = Ts
        self.dt = dt_inner
        self.n_inner = int(Ts / dt_inner)
        self.u_prev = 0.0

    def _predict(self, x0: float, u_seq: np.ndarray, d: float) -> np.ndarray:
        x = x0
        xs = []
        for k in range(self.N):
            for _ in range(self.n_inner):
                sx = math.sqrt(max(x, 0.0))
                x = max(x + self.dt * (u_seq[k] + d - self.K * sx) / self.A,
                        0.0)
            xs.append(x)
        return np.array(xs)

    def query(self, x_curr: float, ysp: float, d: float,
              u_prev: float | None = None) -> float:
        if u_prev is None:
            u_prev = self.u_prev
        # Warm-start within DU_MAX of u_prev so SLSQP starts feasible
        u0 = np.full(self.N, u_prev)

        def cost(u_seq):
            xs = self._predict(x_curr, u_seq, d)
            track = np.sum((xs - ysp) ** 2)
            du = np.diff(np.concatenate([[u_prev], u_seq]))
            effort = self.R * np.sum(du ** 2)
            return track + effort

        bounds = [(U_MIN, U_MAX)] * self.N

        # Rate constraint: |u[k] - u[k-1]| <= DU_MAX (Kardamaki Sec 4.1.1).
        # SLSQP 'ineq' constraints require fun(x) >= 0.
        constraints = [
            {"type": "ineq",
             "fun": lambda u, up=u_prev: DU_MAX - (u[0] - up)},
            {"type": "ineq",
             "fun": lambda u, up=u_prev: DU_MAX - (up - u[0])},
        ]
        for k in range(self.N - 1):
            constraints.append({"type": "ineq",
                "fun": lambda u, k=k: DU_MAX - (u[k+1] - u[k])})
            constraints.append({"type": "ineq",
                "fun": lambda u, k=k: DU_MAX - (u[k] - u[k+1])})

        res = minimize(cost, u0, method="SLSQP", bounds=bounds,
                        constraints=constraints,
                        options={"maxiter": 50, "ftol": 1e-6})
        u_cmd = float(res.x[0])
        # Safety clip in case SLSQP returns a marginally-infeasible point
        u_cmd = float(np.clip(u_cmd, u_prev - DU_MAX, u_prev + DU_MAX))
        u_cmd = float(np.clip(u_cmd, U_MIN, U_MAX))
        self.u_prev = u_cmd
        return u_cmd

    def reset(self):
        self.u_prev = 0.0


# ============================================================
# Closed-loop rollout (one trajectory)
# ============================================================
def rollout_pinn(model: PINN_Controller, x0: float, ysp_fn, d_fn,
                  A: float = A_DEFAULT, K: float = K_VALVE,
                  sim_time: float = 50.0, Ts: float = TS_CTRL,
                  dt: float = DT_PLANT, noise_std: float = 0.0,
                  rng=None) -> dict:
    """Closed-loop PINN-MPC rollout. ysp_fn(t) and d_fn(t) return current
    setpoint and disturbance."""
    rng = rng or np.random.default_rng(0)
    n_steps = int(sim_time / Ts)
    n_inner = int(Ts / dt)
    x = float(x0)
    u_prev = K * math.sqrt(max(x, 0.0))
    t_log = [0.0]
    x_log = [x]
    u_log = [u_prev]
    ysp_log = [ysp_fn(0.0)]
    d_log = [d_fn(0.0)]
    model.eval()
    with torch.no_grad():
        for k in range(n_steps):
            t_curr = k * Ts
            ysp = ysp_fn(t_curr)
            d = d_fn(t_curr)
            x_meas = x + rng.normal(0.0, noise_std) if noise_std > 0 else x
            x_meas = max(x_meas, 0.0)
            t_q = torch.tensor([Ts], device=DEVICE)
            x_t = torch.tensor([x_meas], device=DEVICE, dtype=torch.float32)
            u_t = torch.tensor([u_prev], device=DEVICE, dtype=torch.float32)
            y_t = torch.tensor([ysp], device=DEVICE, dtype=torch.float32)
            d_t = torch.tensor([d], device=DEVICE, dtype=torch.float32)
            _, u_pred = model(t_q, x_t, u_t, y_t, d_t)
            u_cmd = float(u_pred.item())
            u_cmd = float(np.clip(u_cmd, U_MIN, U_MAX))
            for _ in range(n_inner):
                sx = math.sqrt(max(x, 0.0))
                dx_dt = (u_cmd + d - K * sx) / A
                x = max(x + dt * dx_dt, 0.0)
            u_prev = u_cmd
            t_log.append((k + 1) * Ts)
            x_log.append(x)
            u_log.append(u_cmd)
            ysp_log.append(ysp_fn((k + 1) * Ts))
            d_log.append(d_fn((k + 1) * Ts))
    return {"t": np.array(t_log), "x": np.array(x_log),
            "u": np.array(u_log), "ysp": np.array(ysp_log),
            "d": np.array(d_log)}


def rollout_nmpc(nmpc: NMPCOracle, x0: float, ysp_fn, d_fn,
                  A: float = A_DEFAULT, K: float = K_VALVE,
                  sim_time: float = 50.0, Ts: float = TS_CTRL,
                  dt: float = DT_PLANT) -> dict:
    nmpc.reset()
    n_steps = int(sim_time / Ts)
    n_inner = int(Ts / dt)
    x = float(x0)
    u_prev = K * math.sqrt(max(x, 0.0))
    t_log = [0.0]; x_log = [x]; u_log = [u_prev]
    ysp_log = [ysp_fn(0.0)]; d_log = [d_fn(0.0)]
    for k in range(n_steps):
        t_curr = k * Ts
        ysp = ysp_fn(t_curr)
        d = d_fn(t_curr)
        u_cmd = nmpc.query(x, ysp, d, u_prev=u_prev)
        for _ in range(n_inner):
            sx = math.sqrt(max(x, 0.0))
            x = max(x + dt * (u_cmd + d - K * sx) / A, 0.0)
        u_prev = u_cmd
        t_log.append((k + 1) * Ts)
        x_log.append(x); u_log.append(u_cmd)
        ysp_log.append(ysp_fn((k + 1) * Ts))
        d_log.append(d_fn((k + 1) * Ts))
    return {"t": np.array(t_log), "x": np.array(x_log),
            "u": np.array(u_log), "ysp": np.array(ysp_log),
            "d": np.array(d_log)}


# ============================================================
# Scenario definitions (matching Kardamaki Figs 5, 6, 7, 11)
# ============================================================
def fig11_scenario():
    """Fig 11: y0=0.3, ysp=1.4 at t=5s, d=0.2 at t=25s, sim=50s."""
    return {
        "x0": 0.3,
        "ysp_fn": lambda t: 0.3 if t < 5.0 else 1.4,
        "d_fn": lambda t: 0.0 if t < 25.0 else 0.2,
        "sim_time": 50.0,
    }


def fig5_scenario():
    """Fig 5: multi-step setpoint + disturbance changes over 360s.

    Reproduced from Kardamaki Fig 5 by reading the plot:
      SP steps (s, value):  (0, 0.5), (10, 1.4), (70, 0.3), (130, 0.7),
                            (190, 0.5), (250, 0.9), (310, 1.8)
      d steps  (s, value):  (0, 0.0), (40, 0.2), (70, 0.0), (100, 0.15),
                            (130, 0.0), (160, 0.3), (190, 0.0), (280, 0.1),
                            (310, 0.0), (340, 0.4)
    """
    sp_steps = [(0, 0.5), (10, 1.4), (70, 0.3), (130, 0.7), (190, 0.5),
                (250, 0.9), (310, 1.8)]
    d_steps  = [(0, 0.0), (40, 0.2), (70, 0.0), (100, 0.15), (130, 0.0),
                (160, 0.3), (190, 0.0), (280, 0.1), (310, 0.0), (340, 0.4)]
    def _step(t, steps):
        v = steps[0][1]
        for tk, vk in steps:
            if t >= tk: v = vk
            else: break
        return v
    return {
        "x0": 0.5,
        "ysp_fn": lambda t: _step(t, sp_steps),
        "d_fn":   lambda t: _step(t, d_steps),
        "sim_time": 360.0,
    }


def fig7_scenarios():
    """Fig 7: same SP/disturbance as Fig 11, but plant has A=0.5, 1.0, 2.0."""
    base = fig11_scenario()
    return [{"label": f"A={A:.1f} m^2", "A": A, **base}
            for A in (0.5, 1.0, 2.0)]


# ============================================================
# Figure 11 — Single scenario, PINN vs NMPC
# ============================================================
# Curve style table: matches reading order on the plot (top-to-bottom)
PINN_STYLES = {
    "Kardamaki PINN":   {"color": "#7AB8E0", "lw": 1.7, "ls": "--"},
    "LLM-AutoOpt PINN": {"color": "#1F77B4", "lw": 2.2, "ls": "-"},
    "LLM+DPC PINN":     {"color": "#117A1F", "lw": 2.2, "ls": "-"},
}
U_COLOR_PINN = {
    "Kardamaki PINN":   "#B98AC9",
    "LLM-AutoOpt PINN": "#8A4FBE",
    "LLM+DPC PINN":     "#5D2F8C",
}


def make_fig11(models: dict, out_dir, do_nmpc=True):
    """Closed-loop overlay of N PINN variants (+ optional NMPC).

    `models` is an ordered dict {label: PINN_Controller}. Each label gets
    its own line; styles drawn from PINN_STYLES. NMPC is always rendered
    last (dotted) if do_nmpc=True.
    """
    sc = fig11_scenario()
    rolls = {}
    for label, mdl in models.items():
        print(f"[Fig 11] PINN rollout: {label}...")
        rolls[label] = rollout_pinn(mdl, sc["x0"], sc["ysp_fn"], sc["d_fn"],
                                       sim_time=sc["sim_time"])
    nmpc = None
    if do_nmpc:
        print("[Fig 11] NMPC oracle rollout...")
        nmpc = rollout_nmpc(NMPCOracle(), sc["x0"], sc["ysp_fn"], sc["d_fn"],
                              sim_time=sc["sim_time"])

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax_y, ax_u, ax_d = axes

    # --- Controlled Output ---
    # setpoint first (so it sits behind the curves)
    any_r = next(iter(rolls.values()))
    ax_y.plot(any_r["t"], any_r["ysp"], "k--", label=r"$y_{sp}$", lw=1.6)
    for label, r in rolls.items():
        s = PINN_STYLES.get(label, {"color": "tab:blue", "lw": 2, "ls": "-"})
        ax_y.plot(r["t"], r["x"], color=s["color"], lw=s["lw"], ls=s["ls"],
                   label=f"{label} y(t)")
    if nmpc is not None:
        ax_y.plot(nmpc["t"], nmpc["x"], color="navy", lw=1.5, ls=":",
                   label="NMPC y(t) (oracle)")
    ax_y.set_ylabel("Tank Level (m)")
    ax_y.set_title("Controlled Output")
    ax_y.legend(loc="lower right", fontsize=9)
    ax_y.grid(alpha=0.3)

    # --- Control Signal ---
    for label, r in rolls.items():
        c = U_COLOR_PINN.get(label, "purple")
        s = PINN_STYLES.get(label, {"lw": 1.5, "ls": "-"})
        ax_u.plot(r["t"], r["u"], color=c, lw=s["lw"], ls=s["ls"],
                   label=f"{label} u(t)")
    if nmpc is not None:
        ax_u.plot(nmpc["t"], nmpc["u"], color="indigo", lw=1.3, ls=":",
                   label="NMPC u(t)")
    ax_u.set_ylabel("Control Flow (m³/s)")
    ax_u.set_title("Control Signal")
    ax_u.legend(loc="upper right", fontsize=9)
    ax_u.grid(alpha=0.3)

    # --- Disturbance Profile ---
    ax_d.plot(any_r["t"], any_r["d"], "--", color="teal", lw=1.8,
              label="d(t)")
    ax_d.set_ylabel("Disturbance (m³/s)")
    ax_d.set_xlabel("Time (s)")
    ax_d.set_title("Disturbance Profile")
    ax_d.legend(loc="best")
    ax_d.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "fig11_single_scenario.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Fig 11] -> {out_path}")
    return {"rolls": rolls, "nmpc": nmpc}


# ============================================================
# Figure 5 — Multi-step
# ============================================================
def make_fig5(models: dict, out_dir, noise_std=0.0, fig_id="fig05"):
    """Multi-step setpoint + disturbance scenario. Accepts {label: model}."""
    sc = fig5_scenario()
    rolls = {}
    for label, mdl in models.items():
        print(f"[{fig_id}] PINN rollout ({label}, noise={noise_std})...")
        rolls[label] = rollout_pinn(mdl, sc["x0"], sc["ysp_fn"], sc["d_fn"],
                                       sim_time=sc["sim_time"],
                                       noise_std=noise_std)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    ax_y, ax_u, ax_d = axes
    any_r = next(iter(rolls.values()))

    ax_y.plot(any_r["t"], any_r["ysp"], "k--", label=r"$y_{sp}$", lw=1.5)
    for label, r in rolls.items():
        s = PINN_STYLES.get(label, {"color": "tab:blue", "lw": 1.5, "ls": "-"})
        ax_y.plot(r["t"], r["x"], color=s["color"], lw=s["lw"], ls=s["ls"],
                   label=f"{label} y(t)")
    ax_y.set_ylabel("Tank Level (m)")
    ax_y.set_title("Controlled Output")
    ax_y.legend(loc="best", fontsize=9)
    ax_y.grid(alpha=0.3)

    for label, r in rolls.items():
        c = U_COLOR_PINN.get(label, "purple")
        s = PINN_STYLES.get(label, {"lw": 1.2, "ls": "-"})
        ax_u.plot(r["t"], r["u"], color=c, lw=s["lw"], ls=s["ls"],
                   label=f"{label} u(t)")
    ax_u.set_ylabel("Control Flow (m³/s)")
    ax_u.set_title("Control Signal")
    ax_u.legend(loc="best", fontsize=9)
    ax_u.grid(alpha=0.3)

    ax_d.plot(any_r["t"], any_r["d"], "--", color="teal", label="d(t)")
    ax_d.set_ylabel("Disturbance (m³/s)")
    ax_d.set_xlabel("Time (s)")
    ax_d.set_title("Disturbance Profile")
    ax_d.legend(loc="best")
    ax_d.grid(alpha=0.3)

    plt.tight_layout()
    name = "fig05_multistep_tracking" if not noise_std \
        else "fig06_multistep_with_noise"
    out_path = out_dir / f"{name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{fig_id}] -> {out_path}")
    return rolls


# ============================================================
# Figure 7 — Model mismatch (A = 0.5, 1.0, 2.0)
# ============================================================
def make_fig7(models: dict, out_dir):
    """Fig 7 reproduces Kardamaki's model-mismatch sweep (A in {0.5, 1, 2}).

    Uses the BEST model (last entry in dict) by convention - typically
    LLM+DPC PINN. To overlay multiple controllers, call once per model.
    """
    print("[Fig 7] Model-mismatch rollouts (A=0.5, 1.0, 2.0)...")
    # Use last model in the dict (highest-quality variant)
    label, model = list(models.items())[-1]
    print(f"  using controller: {label}")
    sc = fig11_scenario()
    rolls = {}
    for A in (0.5, 1.0, 2.0):
        rolls[A] = rollout_pinn(model, sc["x0"], sc["ysp_fn"], sc["d_fn"],
                                  A=A, sim_time=sc["sim_time"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax_y, ax_u, ax_d = axes
    colors = {"0.5": "#1f77b4", "1.0": "#5fa8d3", "2.0": "#9bcbeb"}
    styles = {"0.5": "-", "1.0": "-.", "2.0": ":"}
    for A in (0.5, 1.0, 2.0):
        r = rolls[A]
        ax_y.plot(r["t"], r["x"], styles[f"{A:.1f}"],
                   label=f"A={A:.1f} m²", lw=2)
        ax_u.plot(r["t"], r["u"], styles[f"{A:.1f}"], color="purple",
                   label=f"A={A:.1f} m²", lw=1.5)
    ax_y.plot(rolls[1.0]["t"], rolls[1.0]["ysp"], "k--",
              label=r"$y_{sp}$", lw=1.5)
    ax_y.set_ylabel("Tank Level (m)")
    ax_y.set_title("Controlled Output")
    ax_y.legend(loc="best")
    ax_y.grid(alpha=0.3)
    ax_u.set_ylabel("Control Flow (m³/s)")
    ax_u.set_title("Control Signal")
    ax_u.legend(loc="best")
    ax_u.grid(alpha=0.3)
    ax_d.plot(rolls[1.0]["t"], rolls[1.0]["d"], "--", color="teal",
              label="d(t)")
    ax_d.set_ylabel("Disturbance (m³/s)")
    ax_d.set_xlabel("Time (s)")
    ax_d.set_title("Disturbance Profile")
    ax_d.legend(loc="best")
    ax_d.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "fig07_model_mismatch.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Fig 7] -> {out_path}")
    return rolls


# ============================================================
# Figure 4 — Training loss curves (Phase 1, Phase 2)
# ============================================================
def make_fig4(hist, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = axes
    ax1.semilogy(hist.get("hist_p1", []), color="#1f77b4")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Total Loss during First Training Phase")
    ax1.grid(alpha=0.3, which="both")
    ax2.semilogy(hist.get("hist_p2", []), color="#2ca02c")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Total Loss during Second Training Phase")
    ax2.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out_path = out_dir / "fig04_loss_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Fig 4] -> {out_path}")


# ============================================================
# Table 10 — Runtime statistics
# ============================================================
def make_table10(model, out_dir, n_episodes: int = 1000,
                  do_nmpc: bool = True):
    """Time PINN forward pass + (optional) NMPC over n_episodes."""
    print(f"[Table 10] timing {n_episodes} PINN forward calls...")
    rng = np.random.default_rng(0)
    pinn_call_ms = []
    pinn_total_ms = []
    model.eval()
    # warm-up
    with torch.no_grad():
        _ = model(torch.tensor([1.0], device=DEVICE),
                   torch.tensor([0.5], device=DEVICE),
                   torch.tensor([0.5], device=DEVICE),
                   torch.tensor([1.0], device=DEVICE),
                   torch.tensor([0.0], device=DEVICE))

    for _ in range(n_episodes):
        x = float(rng.uniform(0.1, 2.0))
        ysp = float(rng.uniform(0.3, 2.0))
        d = float(rng.uniform(0.0, 0.4))
        u_prev = K_VALVE * math.sqrt(max(x, 0.0))
        # 25 calls per episode (sim_time=25, Ts=1)
        t_start_ep = time.perf_counter()
        per_call_times = []
        for _ in range(25):
            t0 = time.perf_counter()
            with torch.no_grad():
                _, _ = model(
                    torch.tensor([TS_CTRL], device=DEVICE),
                    torch.tensor([x], device=DEVICE, dtype=torch.float32),
                    torch.tensor([u_prev], device=DEVICE, dtype=torch.float32),
                    torch.tensor([ysp], device=DEVICE, dtype=torch.float32),
                    torch.tensor([d], device=DEVICE, dtype=torch.float32))
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            per_call_times.append((time.perf_counter() - t0) * 1000.0)
        pinn_call_ms.extend(per_call_times)
        pinn_total_ms.append((time.perf_counter() - t_start_ep) * 1000.0)

    pinn_call_ms = np.array(pinn_call_ms)
    pinn_total_ms = np.array(pinn_total_ms)

    nmpc_call_ms = nmpc_total_ms = None
    if do_nmpc:
        print(f"[Table 10] timing {min(n_episodes, 100)} NMPC episodes "
              f"(NMPC slow - capped at 100)...")
        nmpc_call_ms = []
        nmpc_total_ms = []
        nmpc = NMPCOracle()
        # Only 100 episodes - NMPC takes ~30s per 1000 calls
        for _ in range(min(n_episodes, 100)):
            x = float(rng.uniform(0.1, 2.0))
            ysp = float(rng.uniform(0.3, 2.0))
            d = float(rng.uniform(0.0, 0.4))
            u_prev = K_VALVE * math.sqrt(max(x, 0.0))
            nmpc.reset()
            t_ep = time.perf_counter()
            per_call = []
            for _ in range(25):
                t0 = time.perf_counter()
                nmpc.query(x, ysp, d, u_prev=u_prev)
                per_call.append((time.perf_counter() - t0) * 1000.0)
            nmpc_call_ms.extend(per_call)
            nmpc_total_ms.append((time.perf_counter() - t_ep) * 1000.0)
        nmpc_call_ms = np.array(nmpc_call_ms)
        nmpc_total_ms = np.array(nmpc_total_ms)

    rows = [
        ("Mean time per call (ms)",
         f"{pinn_call_ms.mean():.3f} ± {pinn_call_ms.std():.3f}",
         f"{nmpc_call_ms.mean():.3f} ± {nmpc_call_ms.std():.3f}" if nmpc_call_ms is not None else "N/A",
         f"{KARDAMAKI_TABLE10['pinn_mean_ms']:.3f} ± {KARDAMAKI_TABLE10['pinn_std_ms']:.3f}",
         f"{KARDAMAKI_TABLE10['nmpc_mean_ms']:.3f} ± {KARDAMAKI_TABLE10['nmpc_std_ms']:.3f}"),
        ("Max time per call (ms)",
         f"{pinn_call_ms.max():.3f}",
         f"{nmpc_call_ms.max():.3f}" if nmpc_call_ms is not None else "N/A",
         f"{KARDAMAKI_TABLE10['pinn_max_ms']:.3f}",
         f"{KARDAMAKI_TABLE10['nmpc_max_ms']:.3f}"),
        ("Total per episode (ms)",
         f"{pinn_total_ms.mean():.3f} ± {pinn_total_ms.std():.3f}",
         f"{nmpc_total_ms.mean():.3f} ± {nmpc_total_ms.std():.3f}" if nmpc_total_ms is not None else "N/A",
         f"{KARDAMAKI_TABLE10['pinn_total_ms']:.3f} ± {KARDAMAKI_TABLE10['pinn_total_std']:.3f}",
         f"{KARDAMAKI_TABLE10['nmpc_total_ms']:.3f} ± {KARDAMAKI_TABLE10['nmpc_total_std']:.3f}"),
    ]

    out_path = out_dir / "table10_runtime.csv"
    with open(out_path, "w") as f:
        f.write("metric,Ours PINN-MPC,Ours NMPC,Kardamaki PINN-MPC,Kardamaki NMPC\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"[Table 10] -> {out_path}")

    print("\n=== Table 10: Runtime ===")
    print(f"{'metric':<28}{'Ours PINN':>22}{'Ours NMPC':>26}{'Kard PINN':>22}{'Kard NMPC':>26}")
    for r in rows:
        print(f"{r[0]:<28}{r[1]:>22}{r[2]:>26}{r[3]:>22}{r[4]:>26}")

    speedup = (nmpc_call_ms.mean() / pinn_call_ms.mean()
               if nmpc_call_ms is not None else None)
    if speedup:
        print(f"\nOur PINN is {speedup:.0f}x faster than our NMPC oracle.")
    return {
        "pinn_call_mean_ms": float(pinn_call_ms.mean()),
        "pinn_call_std_ms":  float(pinn_call_ms.std()),
        "pinn_call_max_ms":  float(pinn_call_ms.max()),
        "pinn_total_mean_ms":float(pinn_total_ms.mean()),
        "pinn_total_std_ms": float(pinn_total_ms.std()),
        "nmpc_call_mean_ms": float(nmpc_call_ms.mean()) if nmpc_call_ms is not None else None,
        "nmpc_call_max_ms":  float(nmpc_call_ms.max()) if nmpc_call_ms is not None else None,
        "nmpc_total_mean_ms":float(nmpc_total_ms.mean()) if nmpc_total_ms is not None else None,
        "speedup_vs_nmpc":   speedup,
    }


# ============================================================
# Table 4 — Performance metrics (mean/max/min tracking, disturbance)
# ============================================================
def make_table4(model, out_dir, n_tracking: int = 500,
                  n_disturbance: int = 500):
    print(f"[Table 4] evaluating on Kardamaki test set "
          f"({n_tracking}+{n_disturbance} episodes)...")
    metrics = evaluate_model(model, n_tracking=n_tracking,
                                n_disturbance=n_disturbance,
                                use_kardamaki_samples=True)

    out_path = out_dir / "table04_performance_metrics.csv"
    with open(out_path, "w") as f:
        f.write("metric,Ours,Kardamaki paper\n")
        f.write(f"Mean Tracking Offset (m),{metrics['tracking_mean_offset_m']:.4f},"
                f"{KARDAMAKI_TABLE4['tracking_mean_offset_m']:.4f}\n")
        f.write(f"Max Tracking Offset (m),{metrics['tracking_max_offset_m']:.4f},"
                f"{KARDAMAKI_TABLE4['tracking_max_offset_m']:.4f}\n")
        f.write(f"Mean Disturbance Offset (m),{metrics['disturbance_mean_offset_m']:.4f},"
                f"{KARDAMAKI_TABLE4['disturbance_mean_offset_m']:.4f}\n")
        f.write(f"Max Disturbance Offset (m),{metrics['disturbance_max_offset_m']:.4f},"
                f"{KARDAMAKI_TABLE4['disturbance_max_offset_m']:.4f}\n")
    print(f"[Table 4] -> {out_path}")

    print("\n=== Table 4: Performance Metrics ===")
    print(f"{'metric':<32}{'Ours':>12}{'Kardamaki':>12}{'delta %':>12}")
    for k_ours, k_paper in [
        ("tracking_mean_offset_m",   "tracking_mean_offset_m"),
        ("tracking_max_offset_m",    "tracking_max_offset_m"),
        ("disturbance_mean_offset_m","disturbance_mean_offset_m"),
        ("disturbance_max_offset_m", "disturbance_max_offset_m"),
    ]:
        o = metrics[k_ours]
        p = KARDAMAKI_TABLE4[k_paper]
        d = (o - p) / p * 100
        print(f"{k_ours:<32}{o:>12.4f}{p:>12.4f}{d:>+12.1f}")
    return metrics


# ============================================================
# Table 3 — Loss term decomposition (end of Phase 1 vs end of Phase 2)
# ============================================================
def make_table3(model, hp, x0_all, u0_all, ysp_all, d0_all, out_dir):
    """Compute the breakdown of each loss term on a fresh batch at the
    current model state. We do this AFTER training, so we report only the
    'after Phase 2' column. To get 'after Phase 1' run train_pinn_siso with
    K2=0 and call this function on that model."""
    print("[Table 3] computing per-term loss decomposition...")
    from agentic_pinn_mpc.pinn_siso import generate_collocation_points
    t_wp = torch.cat([torch.arange(0, hp.T_horizon, hp.Ts, device=DEVICE),
                       torch.tensor([hp.T_horizon], device=DEVICE)])
    N_WP = t_wp.shape[0]
    t_col = generate_collocation_points(horizon=hp.T_horizon)
    N_COL = t_col.shape[0]
    bs = min(hp.bs, x0_all.shape[0])
    idx = torch.randperm(x0_all.shape[0])[:bs]
    x0 = x0_all[idx]; u0 = u0_all[idx]; ysp = ysp_all[idx]; d0 = d0_all[idx]
    model.eval()
    # Phase 2 active weights
    loss_total, comps = model.loss(bs, t_wp, N_WP, t_col, N_COL,
                                      x0, u0, ysp, d0,
                                      hp.w_ode, hp.w_ic, hp.w_ytrk, hp.w_utrk,
                                      hp.w_du, hp.w_u, hp.w_x)
    (loss_ode, loss_ytrk, loss_utrk, loss_du, loss_u, loss_x, loss_ic) = comps
    weighted = {
        "L_ode":  float(loss_ode.item()),
        "L_ic":   float(loss_ic.item()),
        "L_ytrk": float(loss_ytrk.item()),
        "L_utrk": float(loss_utrk.item()),
        "L_du":   float(loss_du.item()),
        "L_u":    float(loss_u.item()),
        "L_x":    float(loss_x.item()),
    }
    raw = {  # divide weighted by its weight to recover the pre-weighted scalar
        "L_ode":  weighted["L_ode"]  / hp.w_ode if hp.w_ode > 0 else 0.0,
        "L_ic":   weighted["L_ic"]   / hp.w_ic if hp.w_ic > 0 else 0.0,
        "L_ytrk": weighted["L_ytrk"] / hp.w_ytrk if hp.w_ytrk > 0 else 0.0,
        "L_utrk": weighted["L_utrk"] / hp.w_utrk if hp.w_utrk > 0 else 0.0,
        "L_du":   weighted["L_du"]   / hp.w_du if hp.w_du > 0 else 0.0,
        "L_u":    weighted["L_u"]    / hp.w_u if hp.w_u > 0 else 0.0,
        "L_x":    weighted["L_x"]    / hp.w_x if hp.w_x > 0 else 0.0,
    }

    out_path = out_dir / "table03_loss_decomposition.csv"
    with open(out_path, "w") as f:
        f.write("Term,Raw (pre-weighted),Weighted\n")
        for k in ["L_ode", "L_ic", "L_ytrk", "L_utrk", "L_du", "L_u", "L_x"]:
            f.write(f"{k},{raw[k]:.4e},{weighted[k]:.4e}\n")
        f.write(f"Total (weighted),,{float(loss_total.item()):.4e}\n")
    print(f"[Table 3] -> {out_path}")
    return {"raw": raw, "weighted": weighted,
            "total_weighted": float(loss_total.item())}


# ============================================================
# Main
# ============================================================
def _train_pinn(cfg: dict, K1: int, K2: int, bs: int,
                  x0_all, u0_all, ysp_all, d0_all,
                  label: str = "PINN") -> tuple:
    """Train one PINN with the given cfg dict. Returns (model, hist)."""
    hp_kwargs = {k: v for k, v in cfg.items()
                 if k in PINNHparams.__dataclass_fields__}
    hp = PINNHparams(K1=K1, K2=K2, bs=bs, **hp_kwargs)
    t0 = time.time()
    print(f"\n[Train: {label}] K1={K1}, K2={K2}, bs={bs}...")
    model, hist = train_pinn_siso(hp, x0_all, u0_all, ysp_all, d0_all,
                                     verbose=False)
    print(f"  trained in {time.time()-t0:.1f}s")
    return model, hist, hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-cfg",
                    default="results/llm_phaseA/llm_trials.json",
                    help="Path to llm_trials.json from STEP 2")
    ap.add_argument("--output", default="results/paper_figures/")
    ap.add_argument("--K1", type=int, default=10000)
    ap.add_argument("--K2", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--n-runtime-eps", type=int, default=1000,
                    help="Episodes for Table 10 PINN timing")
    ap.add_argument("--skip-nmpc", action="store_true",
                    help="Skip NMPC oracle in Fig 11 + Table 10")
    ap.add_argument("--skip-kard-baseline", action="store_true",
                    help="Don't train Kardamaki's baseline PINN (faster)")
    ap.add_argument("--skip-dpc", action="store_true",
                    help="Don't apply DPC refinement (only LLM PINN shown)")
    ap.add_argument("--dpc-epochs-track", type=int, default=50)
    ap.add_argument("--dpc-epochs-dist", type=int, default=200)
    ap.add_argument("--only", default="",
                    help="Comma-separated subset: fig4,fig5,fig6,fig7,fig11,"
                         "table3,table4,table10")
    ap.add_argument("--use-saved-model", default="",
                    help="Path to a saved torch model to skip retraining")
    a = ap.parse_args()

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = set(a.only.split(",")) if a.only else {
        "fig4", "fig5", "fig6", "fig7", "fig11",
        "table3", "table4", "table10",
    }

    # ---- Load LLM-best cfg ----
    with open(a.best_cfg) as f:
        llm = json.load(f)
    best_cfg = llm["best_cfg"]
    print(f"\nLLM-best cfg from {a.best_cfg}:")
    for k, v in best_cfg.items():
        print(f"  {k}: {v}")
    print(f"  best_score: {llm.get('best_score', 'N/A')}")

    # ---- Train models (Kardamaki baseline + LLM-AutoOpt + optional DPC) ----
    print("\n=== Training PINN variants for paper figures ===")
    x0_all, u0_all, ysp_all, d0_all = load_training_data()

    models = {}  # ordered dict: label -> PINN_Controller
    hist_llm = None
    hp_llm = None

    # Kardamaki baseline (their published config from cell 19)
    if not a.skip_kard_baseline:
        model_kard, _, _ = _train_pinn(
            dict(KARDAMAKI_BEST), a.K1, a.K2, a.bs,
            x0_all, u0_all, ysp_all, d0_all,
            label="Kardamaki baseline")
        torch.save(model_kard.state_dict(), out_dir / "pinn_kardamaki.pt")
        models["Kardamaki PINN"] = model_kard

    # LLM-AutoOpt best
    if a.use_saved_model and os.path.exists(a.use_saved_model):
        print(f"\nLoading saved LLM-AutoOpt model: {a.use_saved_model}")
        model_llm = PINN_Controller().to(DEVICE)
        model_llm.load_state_dict(torch.load(a.use_saved_model,
                                                map_location=DEVICE))
        hist_llm = {"hist_p1": [], "hist_p2": []}
        hp_llm = PINNHparams(
            **{k: v for k, v in best_cfg.items()
               if k in PINNHparams.__dataclass_fields__},
            K1=a.K1, K2=a.K2, bs=a.bs)
    else:
        model_llm, hist_llm, hp_llm = _train_pinn(
            best_cfg, a.K1, a.K2, a.bs,
            x0_all, u0_all, ysp_all, d0_all,
            label="LLM-AutoOpt best")
        torch.save(model_llm.state_dict(), out_dir / "pinn_llm.pt")
    models["LLM-AutoOpt PINN"] = model_llm

    # Optional DPC refinement on top of LLM-AutoOpt
    if not a.skip_dpc:
        import copy
        model_dpc = copy.deepcopy(model_llm)
        td = (x0_all, u0_all, ysp_all, d0_all)
        print(f"\n=== DPC refinement on LLM-AutoOpt (two-phase, Kardamaki-dist) ===")
        print(f"  Phase 1: tracking-only ({a.dpc_epochs_track} ep, lr=1e-5)")
        cfg_track = DPCRefineCfg(epochs=a.dpc_epochs_track, bs=32, lr=1e-5,
                                    mode="tracking")
        model_dpc, _ = refine_with_dpc(model_dpc, cfg_track, verbose=False,
                                          train_data=td)
        print(f"  Phase 2: disturbance-only ({a.dpc_epochs_dist} ep, "
              f"lr=5e-6, oversample=1.5)")
        cfg_dist = DPCRefineCfg(epochs=a.dpc_epochs_dist, bs=32, lr=5e-6,
                                   mode="disturbance",
                                   disturbance_oversample=1.5)
        model_dpc, _ = refine_with_dpc(model_dpc, cfg_dist, verbose=False,
                                          train_data=td)
        torch.save(model_dpc.state_dict(), out_dir / "pinn_llm_dpc.pt")
        models["LLM+DPC PINN"] = model_dpc

    summary = {
        "llm_best_cfg": best_cfg,
        "llm_best_score": llm.get("best_score"),
        "models_trained": list(models.keys()),
    }

    # ---- Generate outputs ----
    if "fig4" in targets and hist_llm and hist_llm.get("hist_p1"):
        make_fig4(hist_llm, out_dir)
        summary["fig4_saved"] = True

    if "fig11" in targets:
        roll = make_fig11(models, out_dir, do_nmpc=not a.skip_nmpc)
        summary["fig11_final_y"] = {
            label: float(r["x"][-1]) for label, r in roll["rolls"].items()}
        if roll["nmpc"]:
            summary["fig11_nmpc_final_y"] = float(roll["nmpc"]["x"][-1])

    if "fig5" in targets:
        rolls = make_fig5(models, out_dir, noise_std=0.0, fig_id="fig05")
        summary["fig5_final_y"] = {label: float(r["x"][-1])
                                      for label, r in rolls.items()}

    if "fig6" in targets:
        rolls = make_fig5(models, out_dir, noise_std=0.02, fig_id="fig06")
        summary["fig6_final_y"] = {label: float(r["x"][-1])
                                      for label, r in rolls.items()}

    if "fig7" in targets:
        rolls = make_fig7(models, out_dir)
        summary["fig7_final_y_byA"] = {f"A={A}": float(rolls[A]["x"][-1])
                                          for A in (0.5, 1.0, 2.0)}

    # Best (DPC if present, else LLM) goes into eval tables for paper claims
    best_label, best_model = list(models.items())[-1]
    print(f"\n=== Tables use BEST model: {best_label} ===")

    if "table4" in targets:
        m4 = make_table4(best_model, out_dir,
                            n_tracking=500, n_disturbance=500)
        summary["table4_best_label"] = best_label
        summary["table4"] = m4

    if "table10" in targets:
        m10 = make_table10(best_model, out_dir,
                              n_episodes=a.n_runtime_eps,
                              do_nmpc=not a.skip_nmpc)
        summary["table10"] = m10

    if "table3" in targets:
        m3 = make_table3(best_model, hp_llm,
                            x0_all, u0_all, ysp_all, d0_all, out_dir)
        summary["table3"] = m3

    # ---- Save summary ----
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n=== All paper figures + tables saved to {out_dir} ===")
    print(f"  summary.json includes all numeric results")


if __name__ == "__main__":
    main()
