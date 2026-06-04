"""paper_figures_fourtank_distill.py - Bloor Fig 7 & 8 style figures for our methods.

Generates two figures mirroring Bloor 2025 §4.3.4 (PC-Gym four-tank Fig 7 + 8):

  fig_fourtank_distill_tracking.{png,pdf}
    4-panel: h_1, h_2, v_1, v_2 over time. Median + min-max bands across
    `n_reps` episodes. Black dashed reference setpoint trajectory.

  fig_fourtank_distill_rewards.{png,pdf}
    Cumulative reward histograms across `n_reps` episodes for all methods.
    Oracle plotted as vertical dashed line (Bloor's convention).

Setpoint protocol = Bloor Fig 7 EXACT:
  h_1_sp: 0.5 -> 0.1 (step down at t=500s)
  h_2_sp: 0.2 -> 0.3 (step up   at t=500s)

Usage:
  !python -u paper_figures_fourtank_distill.py \\
      --pinn-distill /content/drive/MyDrive/pinn_mpc_results/nmpc_distill_fourtank_seed0/pinn_fourtank_distill_pretrain.pt \\
      --pinn-llm-dpc /content/drive/MyDrive/pinn_mpc_results/two_phase_dpc_fourtank/pinn_fourtank_p2_hard.pt \\
      --sac          /content/drive/MyDrive/pinn_mpc_results/rl_baselines_fourtank/sac_fourtank.zip \\
      --ddpg         /content/drive/MyDrive/pinn_mpc_results/rl_baselines_fourtank/ddpg_fourtank.zip \\
      --n-reps 50 \\
      --output-dir /content/drive/MyDrive/pinn_mpc_results/paper_figures_distill

All --* model args are OPTIONAL; missing models are skipped (Oracle and the
proposed PINN+Distill are always plotted if --pinn-distill is given).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from agentic_pcgym.pinn_fourtank import PINN_FourTank, DEVICE
from agentic_pcgym.evaluator import closed_loop_fourtank
from agentic_pcgym.nmpc_fourtank import FourTankNMPC, FourTankOperatingPoint
from agentic_pcgym.plants.fourtank import FourTankScenario


def make_bloor_paper_schedule(n_steps: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Bloor Fig 7 EXACT setpoint schedule (deterministic, no jitter)."""
    h1 = np.zeros(n_steps, dtype=float)
    h2 = np.zeros(n_steps, dtype=float)
    k_step = n_steps // 2   # t=500s (n_steps=60, dt=16.67s, so k=30 → t=500s)
    h1[:k_step] = 0.5
    h1[k_step:] = 0.1
    h2[:k_step] = 0.2
    h2[k_step:] = 0.3
    return h1, h2


def rollout(query_fn, n_reps: int, seed: int, n_steps: int = 60) -> dict:
    """Run n_reps episodes of one controller. Returns trajectories + cum rewards."""
    rng = np.random.default_rng(seed)
    op = FourTankOperatingPoint()
    h1_sched, h2_sched = make_bloor_paper_schedule(n_steps)

    out = {k: np.zeros((n_reps, n_steps)) for k in ["h1","h2","h3","h4","v1","v2"]}
    out["rewards"] = np.zeros(n_reps)
    out["h1_ref"] = h1_sched
    out["h2_ref"] = h2_sched

    for rep in range(n_reps):
        x0 = np.array([op.h_1_0, op.h_2_0, op.h_3_0, op.h_4_0])
        x0 = x0 * rng.uniform(0.8, 1.2, 4)
        result = closed_loop_fourtank(query_fn, x0, h1_sched, h2_sched)
        traj = np.array(result.h_trajectory)   # (n_steps, 4)
        vtraj = np.array(result.v_trajectory)  # (n_steps, 2)
        out["h1"][rep] = traj[:, 0]
        out["h2"][rep] = traj[:, 1]
        out["h3"][rep] = traj[:, 2]
        out["h4"][rep] = traj[:, 3]
        out["v1"][rep] = vtraj[:, 0]
        out["v2"][rep] = vtraj[:, 1]
        out["rewards"][rep] = result.cumulative_reward
    return out


def pinn_query_factory(net):
    @torch.no_grad()
    def q(h1, h2, h3, h4, h1_sp, h2_sp, v1_p, v2_p):
        z = lambda v: torch.tensor([float(v)], device=DEVICE)
        _, _, _, _, v1, v2 = net(
            z(1.0), z(h1), z(h2), z(h3), z(h4),
            z(h1_sp), z(h2_sp), z(v1_p), z(v2_p))
        return (float(v1.item()), float(v2.item()))
    return q


def nmpc_query_factory():
    oracle = FourTankNMPC()
    def q(h1, h2, h3, h4, h1_sp, h2_sp, v1_p, v2_p):
        u = oracle.query(np.array([h1, h2, h3, h4]),
                          sp_h1=h1_sp, sp_h2=h2_sp,
                          u_warm=(v1_p, v2_p))
        return u if u is not None else (v1_p, v2_p)
    return q


def sb3_query_factory(model):
    def q(h1, h2, h3, h4, h1_sp, h2_sp, v1_p, v2_p):
        obs = np.array([h1, h2, h3, h4, h1_sp, h2_sp, v1_p, v2_p],
                        dtype=np.float32)
        a, _ = model.predict(obs, deterministic=True)
        return (float(a[0]), float(a[1]))
    return q


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
# Bloor Fig 7 color palette
COLORS = {
    "Oracle":               "tab:orange",
    "PINN + NMPC Distill":  "tab:green",   # proposed (highlight)
    "LLM + DPC PINN":       "tab:purple",  # ablation
    "Untuned PINN":         "tab:red",     # baseline
    "SAC":                  "crimson",     # RL baselines
    "DDPG":                 "seagreen",
    "PPO":                  "steelblue",
}

LINE_ORDER = [
    "Oracle",
    "PINN + NMPC Distill",
    "LLM + DPC PINN",
    "Untuned PINN",
    "SAC",
    "DDPG",
    "PPO",
]


def plot_tracking_fig7(results: dict, output_path: Path,
                       n_steps: int = 60, dt_s: float = 1000.0/60.0):
    """Bloor Fig 7-style 4-panel: h1, h2, v1, v2 vs time. Median + min-max shading."""
    t = np.arange(n_steps) * dt_s
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

    panels = [
        ("h1", r"$h_1$ [m]"),
        ("h2", r"$h_2$ [m]"),
        ("v1", r"$v_1$ [V]"),
        ("v2", r"$v_2$ [V]"),
    ]

    for ax_idx, (var, ylabel) in enumerate(panels):
        ax = axes[ax_idx]
        # Plot in fixed order, only methods that exist
        for label in LINE_ORDER:
            if label not in results:
                continue
            data = results[label][var]
            med = np.median(data, axis=0)
            lo  = data.min(axis=0)
            hi  = data.max(axis=0)
            color = COLORS.get(label, "tab:blue")
            ds = "steps-post" if var.startswith("v") else None
            ax.plot(t, med, label=label, color=color, linewidth=2.0,
                    drawstyle=ds)
            ax.fill_between(t, lo, hi, color=color, alpha=0.15)

        # Reference dashed line for the controlled variables
        if var == "h1":
            any_res = next(iter(results.values()))
            ax.plot(t, any_res["h1_ref"], "k--", linewidth=1.5,
                    label="Reference", drawstyle="steps-post")
        elif var == "h2":
            any_res = next(iter(results.values()))
            ax.plot(t, any_res["h2_ref"], "k--", linewidth=1.5,
                    label="Reference", drawstyle="steps-post")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    # Single legend, top of figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                bbox_to_anchor=(0.5, 1.06), frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.savefig(str(output_path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  saved {output_path}")


def plot_reward_histogram_fig8(results: dict, output_path: Path):
    """Bloor Fig 8-style cumulative reward histogram. Oracle as vertical dashed line."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Find a common bin range across all non-Oracle methods
    all_rewards = []
    for label in LINE_ORDER:
        if label in results and label != "Oracle":
            all_rewards.extend(results[label]["rewards"].tolist())
    if all_rewards:
        rmin, rmax = min(all_rewards), max(all_rewards)
    else:
        rmin, rmax = -1.0, 0.0
    bins = np.linspace(rmin * 1.05, rmax * 0.95, 20)

    for label in LINE_ORDER:
        if label not in results:
            continue
        rewards = results[label]["rewards"]
        color = COLORS.get(label, "tab:blue")
        if label == "Oracle":
            med = float(np.median(rewards))
            ax.axvline(med, color=color, linestyle="--", linewidth=2.5,
                        label=f"Oracle (med={med:.4f})")
        else:
            med = float(np.median(rewards))
            ax.hist(rewards, bins=bins, alpha=0.55, color=color,
                     label=f"{label} (med={med:.4f})", edgecolor="black",
                     linewidth=0.5)

    ax.set_xlabel("Cumulative Reward")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.savefig(str(output_path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  saved {output_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinn-distill", default=None,
                     help="Path to NMPC-distilled PINN .pt (PROPOSED METHOD)")
    ap.add_argument("--pinn-llm-dpc", default=None,
                     help="Path to LLM+DPC PINN .pt (ABLATION)")
    ap.add_argument("--pinn-untuned", default=None,
                     help="Path to untuned PINN .pt (BASELINE)")
    ap.add_argument("--sac", default=None, help="Path to SAC .zip")
    ap.add_argument("--ddpg", default=None, help="Path to DDPG .zip")
    ap.add_argument("--ppo", default=None, help="Path to PPO .zip")
    ap.add_argument("--n-reps", type=int, default=50,
                     help="Episodes per method (50 matches Bloor)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="results/paper_figures_fourtank_distill")
    a = ap.parse_args()

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Oracle (always)
    print("=== Rollout 50 episodes of NMPC Oracle ===")
    results["Oracle"] = rollout(nmpc_query_factory(), a.n_reps, a.seed)

    # 2. PINN + NMPC Distillation (the PROPOSED method)
    if a.pinn_distill and Path(a.pinn_distill).exists():
        print(f"=== Rollout PINN + NMPC Distill ({a.pinn_distill}) ===")
        net = PINN_FourTank().to(DEVICE)
        net.load_state_dict(torch.load(a.pinn_distill, map_location=DEVICE))
        net.eval()
        results["PINN + NMPC Distill"] = rollout(
            pinn_query_factory(net), a.n_reps, a.seed)

    # 3. LLM + DPC PINN (ablation)
    if a.pinn_llm_dpc and Path(a.pinn_llm_dpc).exists():
        print(f"=== Rollout LLM + DPC PINN ===")
        net = PINN_FourTank().to(DEVICE)
        net.load_state_dict(torch.load(a.pinn_llm_dpc, map_location=DEVICE))
        net.eval()
        results["LLM + DPC PINN"] = rollout(
            pinn_query_factory(net), a.n_reps, a.seed)

    # 4. Untuned PINN (baseline)
    if a.pinn_untuned and Path(a.pinn_untuned).exists():
        print(f"=== Rollout Untuned PINN ===")
        net = PINN_FourTank().to(DEVICE)
        net.load_state_dict(torch.load(a.pinn_untuned, map_location=DEVICE))
        net.eval()
        results["Untuned PINN"] = rollout(
            pinn_query_factory(net), a.n_reps, a.seed)

    # 5/6/7. RL baselines (optional)
    for name, path in [("SAC", a.sac), ("DDPG", a.ddpg), ("PPO", a.ppo)]:
        if path and Path(path).exists():
            print(f"=== Rollout {name} ===")
            if name == "SAC":
                from stable_baselines3 import SAC; model = SAC.load(path)
            elif name == "DDPG":
                from stable_baselines3 import DDPG; model = DDPG.load(path)
            else:
                from stable_baselines3 import PPO; model = PPO.load(path)
            results[name] = rollout(sb3_query_factory(model), a.n_reps, a.seed)

    # Plot
    print("\n=== Generating figures ===")
    plot_tracking_fig7(results,
                        out_dir / "fig_fourtank_distill_tracking.png")
    plot_reward_histogram_fig8(results,
                                 out_dir / "fig_fourtank_distill_rewards.png")

    # Save numeric summary
    summary = {label: {
        "median_reward": float(np.median(r["rewards"])),
        "min_reward":    float(r["rewards"].min()),
        "max_reward":    float(r["rewards"].max()),
        "mean_reward":   float(r["rewards"].mean()),
        "std_reward":    float(r["rewards"].std()),
        "n_reps":        int(len(r["rewards"])),
    } for label, r in results.items()}
    with open(out_dir / "reward_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    print("\n=== Cumulative reward summary (Bloor Fig 7+8 setup) ===")
    print(f"{'Method':<30}{'median':>12}{'min':>12}{'max':>12}{'std':>10}")
    print("-" * 76)
    for label in LINE_ORDER:
        if label not in summary:
            continue
        s = summary[label]
        print(f"{label:<30}{s['median_reward']:>12.4f}"
              f"{s['min_reward']:>12.4f}{s['max_reward']:>12.4f}"
              f"{s['std_reward']:>10.4f}")

    # Opt-gap implied from these rewards (single-protocol headline)
    if "Oracle" in summary and "PINN + NMPC Distill" in summary:
        o = summary["Oracle"]["median_reward"]
        p = summary["PINN + NMPC Distill"]["median_reward"]
        gap = (o - p) / 2.0    # N_e=2 for Bloor step protocol
        print(f"\n>>> PINN+Distill optimality gap (Bloor protocol, N_e=2): "
              f"{gap:.4f}  <<<")

    print(f"\nAll outputs saved to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
