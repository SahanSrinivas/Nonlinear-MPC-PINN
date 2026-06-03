"""paper_figures_fourtank.py - Generates paper figures + tables for the
PC-Gym four-tank MIMO case study (Bloor 2025 Case Study 4 / Johansson 2000).

Outputs (in --output dir):
  fig_fourtank_tracking.png    Closed-loop h_1, h_2 tracking trajectory
                                (Bloor baseline + LLM PINN + LLM+DPC + NMPC)
  fig_fourtank_controls.png    v_1, v_2 control signals over same trajectory
  fig_fourtank_robustness.png  Closed-loop under model mismatch (A_i +- 30%)
  fig_fourtank_noise.png       Closed-loop with measurement noise
  table_fourtank_metrics.csv   Optimality gap + MAD across all PINN stages
  summary_fourtank.json        All numeric results

Assumes the following are already on disk (from earlier runs):
  results/pcgym_fourtank/fourtank/llm.json
       (LLM-AutoOpt best cfg from agentic_pcgym.bench)
  results/two_phase_dpc_fourtank/pinn_fourtank_llm_pretrain.pt   (optional)
  results/two_phase_dpc_fourtank/pinn_fourtank_p2_hard.pt        (optional)

If the .pt files aren't present, this script trains all three PINNs from scratch.
Estimated runtime on T4: ~12 min (3 PINNs trained from scratch) or ~3 min
(if .pt files reused).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from agentic_pcgym.pinn_fourtank import (PINN_FourTank, FourTankPINNHparams,
                                            DEVICE, V_LO, V_HI)
from agentic_pcgym.pinn_training import train_pinn_fourtank
from agentic_pcgym.data_gen import sample_fourtank_episodes
from agentic_pcgym.evaluator import (evaluate_fourtank, closed_loop_fourtank)
from agentic_pcgym.rl_refine import refine_fourtank_dpc, DPCRefineCfg
from agentic_pcgym.nmpc_fourtank import (FourTankNMPC, FourTankOperatingPoint,
                                            FourTankBounds)
from agentic_pcgym.plants.fourtank import FourTankScenario, FourTankParams


# Bloor's default published config (close to literature)
BLOOR_BASELINE_CFG = {
    "w_ode": 100.0, "w_ic": 10.0, "w_ytrk": 10.0, "w_xtrk": 1.0,
    "w_utrk": 1.0, "w_du": 1.0, "w_u": 100.0,
    "lr1": 1e-3, "lr2": 2e-4,
}

# Colour/style scheme matching the SISO paper figures
STYLES = {
    "Bloor PINN":   {"color": "#7AB8E0", "lw": 1.7, "ls": "--"},
    "LLM PINN":     {"color": "#1F77B4", "lw": 2.2, "ls": "-"},
    "LLM+DPC PINN": {"color": "#117A1F", "lw": 2.2, "ls": "-"},
    "NMPC oracle":  {"color": "navy",    "lw": 1.5, "ls": ":"},
}


def _make_query(net):
    @torch.no_grad()
    def q(h1, h2, h3, h4, h1_sp, h2_sp, v1_p, v2_p):
        z = lambda v: torch.tensor([float(v)], device=DEVICE)
        _,_,_,_,v1,v2 = net(z(1.0), z(h1), z(h2), z(h3), z(h4),
                              z(h1_sp), z(h2_sp), z(v1_p), z(v2_p))
        return (float(v1.item()), float(v2.item()))
    return q


def _nmpc_query():
    oracle = FourTankNMPC()
    def q(h1, h2, h3, h4, h1_sp, h2_sp, v1_p, v2_p):
        return oracle.query(np.array([h1, h2, h3, h4]),
                              sp_h1=h1_sp, sp_h2=h2_sp,
                              u_warm=(v1_p, v2_p))
    return q


def _train_baseline(cfg, episodes, K1, K2, bs, label, seed=0):
    print(f"\n=== Train {label} (K={K1},{K2}, bs={bs}) ===")
    hp = FourTankPINNHparams(K1=K1, K2=K2, bs=bs, **{k: float(cfg[k])
        for k in ["w_ode","w_ic","w_ytrk","w_xtrk","w_utrk","w_du","w_u",
                   "lr1","lr2"]})
    net = PINN_FourTank().to(DEVICE)
    t0 = time.time()
    hist = train_pinn_fourtank(net, episodes, hp, verbose=False, seed=seed)
    print(f"  trained in {time.time()-t0:.1f}s")
    return net, hist


def _scenario(h1_sp=0.4, h2_sp=0.3, x0=None):
    if x0 is None:
        op = FourTankOperatingPoint()
        x0 = np.array([op.h_1_0, op.h_2_0, op.h_3_0, op.h_4_0])
    return {"x0": x0, "h1_sp": h1_sp, "h2_sp": h2_sp}


def _rollout(q, sc, params=None, n_steps=None):
    r = closed_loop_fourtank(q, sc["x0"], sc["h1_sp"], sc["h2_sp"],
                                n_steps=n_steps, params=params)
    n = len(r.h_trajectory)
    t = np.arange(1, n + 1) * (1000.0 / 60.0)   # s
    h = np.array(r.h_trajectory)
    v = np.array(r.v_trajectory)
    return {"t": t, "h": h, "v": v,
             "h1_sp": sc["h1_sp"], "h2_sp": sc["h2_sp"],
             "reward": r.cumulative_reward}


def _make_tracking_fig(rolls, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_h, ax_v = axes
    any_r = next(iter(rolls.values()))
    n = len(any_r["t"])
    ax_h.axhline(any_r["h1_sp"], color="black", ls="--", lw=1.3,
                 label=r"$h_{1,sp}$")
    ax_h.axhline(any_r["h2_sp"], color="gray", ls="--", lw=1.3,
                 label=r"$h_{2,sp}$")
    for label, r in rolls.items():
        s = STYLES.get(label, {"color": "tab:blue", "lw": 2, "ls": "-"})
        ax_h.plot(r["t"], r["h"][:, 0], color=s["color"], lw=s["lw"],
                   ls=s["ls"], label=f"{label} h_1(t)")
        ax_h.plot(r["t"], r["h"][:, 1], color=s["color"], lw=s["lw"]*0.7,
                   ls=s["ls"], alpha=0.6, label=f"{label} h_2(t)")
    ax_h.set_ylabel("Tank level (m)")
    ax_h.set_title("Four-tank MIMO closed-loop tracking")
    ax_h.legend(loc="best", fontsize=8, ncol=2)
    ax_h.grid(alpha=0.3)

    for label, r in rolls.items():
        s = STYLES.get(label, {"color": "tab:blue", "lw": 2, "ls": "-"})
        ax_v.plot(r["t"], r["v"][:, 0], color=s["color"], lw=s["lw"],
                   ls=s["ls"], label=f"{label} v_1(t)")
        ax_v.plot(r["t"], r["v"][:, 1], color=s["color"], lw=s["lw"]*0.7,
                   ls=s["ls"], alpha=0.6, label=f"{label} v_2(t)")
    ax_v.set_ylabel("Pump voltage (V)")
    ax_v.set_xlabel("Time (s)")
    ax_v.set_title("Control signals")
    ax_v.legend(loc="best", fontsize=8, ncol=2)
    ax_v.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-cfg",
                    default="results/pcgym_fourtank/fourtank/llm.json")
    ap.add_argument("--dpc-model",
                    default="results/two_phase_dpc_fourtank/pinn_fourtank_p2_hard.pt")
    ap.add_argument("--llm-model",
                    default="results/two_phase_dpc_fourtank/pinn_fourtank_llm_pretrain.pt")
    ap.add_argument("--output", default="results/paper_figures_fourtank/")
    ap.add_argument("--K1", type=int, default=10000)
    ap.add_argument("--K2", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n-train-eps", type=int, default=3000)
    ap.add_argument("--n-eval-reps", type=int, default=20)
    ap.add_argument("--skip-nmpc", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Don't train Bloor's published baseline")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(a.best_cfg) as f:
        llm = json.load(f)
    best_cfg = llm["best_cfg"]
    print(f"LLM-best cfg: {best_cfg}\n")

    print(f"=== Sampling {a.n_train_eps} training episodes... ===")
    t0 = time.time()
    episodes = sample_fourtank_episodes(N=a.n_train_eps, seed=a.seed,
                                            query_nmpc=False, verbose=False)
    print(f"  done in {time.time()-t0:.1f}s\n")

    models = {}

    # 1. Bloor baseline (their published default config)
    if not a.skip_baseline:
        net_b, _ = _train_baseline(BLOOR_BASELINE_CFG, episodes,
                                       a.K1, a.K2, a.bs,
                                       "Bloor baseline PINN", seed=a.seed)
        torch.save(net_b.state_dict(), out_dir / "pinn_fourtank_baseline.pt")
        models["Bloor PINN"] = net_b

    # 2. LLM-AutoOpt PINN
    if Path(a.llm_model).exists():
        print(f"\nLoading saved LLM PINN: {a.llm_model}")
        net_l = PINN_FourTank().to(DEVICE)
        net_l.load_state_dict(torch.load(a.llm_model, map_location=DEVICE))
    else:
        net_l, _ = _train_baseline(best_cfg, episodes, a.K1, a.K2, a.bs,
                                       "LLM-AutoOpt PINN", seed=a.seed)
        torch.save(net_l.state_dict(), out_dir / "pinn_fourtank_llm.pt")
    models["LLM PINN"] = net_l

    # 3. LLM+DPC PINN
    if Path(a.dpc_model).exists():
        print(f"\nLoading saved LLM+DPC PINN: {a.dpc_model}")
        net_d = PINN_FourTank().to(DEVICE)
        net_d.load_state_dict(torch.load(a.dpc_model, map_location=DEVICE))
        models["LLM+DPC PINN"] = net_d
    else:
        print("\n=== Running two-phase DPC on LLM PINN ===")
        net_d = copy.deepcopy(net_l)
        cfg_easy = DPCRefineCfg(epochs=50, bs=32, lr=1e-5, mode="easy",
                                  seed=a.seed)
        refine_fourtank_dpc(net_d, cfg_easy, verbose=False,
                              train_data=episodes)
        cfg_hard = DPCRefineCfg(epochs=150, bs=32, lr=5e-6, mode="hard",
                                  seed=a.seed + 1, early_stop_patience=30)
        refine_fourtank_dpc(net_d, cfg_hard, verbose=False,
                              train_data=episodes)
        torch.save(net_d.state_dict(), out_dir / "pinn_fourtank_dpc.pt")
        models["LLM+DPC PINN"] = net_d

    # ---- Generate trajectories for figures ----
    sc = _scenario(h1_sp=0.45, h2_sp=0.30)
    rolls = {}
    for label, net in models.items():
        print(f"[Rollout] {label} ...")
        rolls[label] = _rollout(_make_query(net), sc)
    if not a.skip_nmpc:
        print("[Rollout] NMPC oracle ...")
        rolls["NMPC oracle"] = _rollout(_nmpc_query(), sc)

    _make_tracking_fig(rolls, out_dir / "fig_fourtank_tracking.png")

    # ---- Model mismatch ----
    print("\n=== Model-mismatch figure (a_i scaled by 0.7, 1.0, 1.3) ===")
    sc_mm = _scenario(h1_sp=0.45, h2_sp=0.30)
    rolls_mm = {}
    p_default = FourTankParams()
    for scale, label in [(0.7, "a_i x0.7"), (1.0, "a_i x1.0"), (1.3, "a_i x1.3")]:
        p = FourTankParams(
            g_a=p_default.g_a, gamma_1=p_default.gamma_1,
            gamma_2=p_default.gamma_2, k_1=p_default.k_1, k_2=p_default.k_2,
            a_1=p_default.a_1 * scale, a_2=p_default.a_2 * scale,
            a_3=p_default.a_3 * scale, a_4=p_default.a_4 * scale,
            A_1=p_default.A_1, A_2=p_default.A_2,
            A_3=p_default.A_3, A_4=p_default.A_4)
        # Use the BEST controller (LLM+DPC) under model mismatch
        rolls_mm[label] = _rollout(_make_query(models["LLM+DPC PINN"]), sc_mm,
                                       params=p)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_h, ax_v = axes
    ax_h.axhline(0.45, color="black", ls="--", lw=1.2, label=r"$h_{1,sp}$")
    ax_h.axhline(0.30, color="gray",  ls="--", lw=1.2, label=r"$h_{2,sp}$")
    cmap = {"a_i x0.7": "#1f77b4", "a_i x1.0": "#117A1F", "a_i x1.3": "#d62728"}
    for label, r in rolls_mm.items():
        ax_h.plot(r["t"], r["h"][:, 0], color=cmap[label], lw=1.8,
                   label=f"{label} h_1(t)")
        ax_h.plot(r["t"], r["h"][:, 1], color=cmap[label], lw=1.4, alpha=0.6,
                   ls="--", label=f"{label} h_2(t)")
        ax_v.plot(r["t"], r["v"][:, 0], color=cmap[label], lw=1.6,
                   label=f"{label} v_1(t)")
        ax_v.plot(r["t"], r["v"][:, 1], color=cmap[label], lw=1.2, alpha=0.6,
                   ls="--", label=f"{label} v_2(t)")
    ax_h.set_ylabel("Tank level (m)"); ax_h.grid(alpha=0.3)
    ax_h.set_title("Robustness under valve-coefficient mismatch (LLM+DPC PINN)")
    ax_h.legend(fontsize=8, ncol=2, loc="best")
    ax_v.set_ylabel("Pump voltage (V)"); ax_v.set_xlabel("Time (s)")
    ax_v.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_fourtank_robustness.png", dpi=150)
    plt.close(fig)
    print(f"  -> {out_dir/'fig_fourtank_robustness.png'}")

    # ---- Metrics table ----
    print("\n=== Computing metrics table ===")
    metrics_log = {}
    for label, net in models.items():
        print(f"  evaluating {label}...")
        metrics_log[label] = evaluate_fourtank(
            _make_query(net), n_reps=a.n_eval_reps, seed=42, verbose=False)
    with (out_dir / "table_fourtank_metrics.csv").open("w") as f:
        f.write("controller,median_reward,optimality_gap,MAD\n")
        for label, m in metrics_log.items():
            f.write(f"{label},{m['median_reward_pi']:.4f},"
                    f"{m['optimality_gap']:.4f},{m['MAD']:.4f}\n")
    print(f"  -> {out_dir/'table_fourtank_metrics.csv'}")

    print("\n=== Final metrics ===")
    print(f"{'controller':<25}{'median R':>15}{'opt gap':>12}{'MAD':>10}")
    for label, m in metrics_log.items():
        print(f"  {label:<25}{m['median_reward_pi']:>15.4f}"
              f"{m['optimality_gap']:>12.4f}{m['MAD']:>10.4f}")

    summary = {
        "case": "fourtank",
        "llm_best_cfg": best_cfg,
        "metrics": {label: {k: float(m[k]) for k in
                              ("median_reward_pi", "median_reward_oracle",
                               "optimality_gap", "MAD")}
                       for label, m in metrics_log.items()},
    }
    with (out_dir / "summary_fourtank.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nAll outputs saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
