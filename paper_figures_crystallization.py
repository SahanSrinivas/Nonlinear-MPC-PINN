"""paper_figures_crystallization.py - Generates paper figures + tables for
the PC-Gym K2SO4 crystallization case study (Bloor 2025 Case Study 3).

Outputs (in --output dir):
  fig_cryst_tracking.png         CV(t), Ln(t), Tc(t) closed-loop trajectory
                                  with all PINN variants + NMPC oracle
  fig_cryst_robustness.png       Closed-loop under kinetic-parameter
                                  mismatch (k_a +- 30%)
  fig_cryst_setpoint_sweep.png   Performance across CV/L_n setpoint grid
  table_cryst_metrics.csv        Optimality gap + MAD across PINN stages
  summary_cryst.json             All numeric results

Assumes from earlier runs:
  results/pcgym_crystallization/crystallization/llm.json
  results/two_phase_dpc_crystallization/pinn_cryst_llm_pretrain.pt   (optional)
  results/two_phase_dpc_crystallization/pinn_cryst_p2_hard.pt        (optional)

Estimated runtime on T4: ~15 min (3 PINNs from scratch) or ~4 min (cached).
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

from agentic_pcgym.pinn_crystallization import (
    PINN_Crystallization, CrystPINNHparams, DEVICE, T_C_LO, T_C_HI)
from agentic_pcgym.pinn_training import train_pinn_crystallization
from agentic_pcgym.data_gen import sample_crystallization_episodes
from agentic_pcgym.evaluator import (evaluate_crystallization,
                                        closed_loop_crystallization)
from agentic_pcgym.rl_refine import refine_crystallization_dpc, DPCRefineCfg
from agentic_pcgym.nmpc_crystallization import (
    CrystallizationNMPC, CrystOperatingPoint)
from agentic_pcgym.plants.crystallization import (CrystParams, CrystScenario)


# Bloor default config (literature-cited starting point)
BLOOR_BASELINE_CFG = {
    "w_ode": 100.0, "w_ic": 10.0, "w_ytrk": 10.0, "w_utrk": 1.0,
    "w_du": 1.0, "w_u": 100.0, "w_x": 10.0,
    "lr1": 1e-3, "lr2": 2e-4,
}

STYLES = {
    "Bloor PINN":   {"color": "#7AB8E0", "lw": 1.7, "ls": "--"},
    "LLM PINN":     {"color": "#1F77B4", "lw": 2.2, "ls": "-"},
    "LLM+DPC PINN": {"color": "#117A1F", "lw": 2.2, "ls": "-"},
    "NMPC oracle":  {"color": "navy",    "lw": 1.5, "ls": ":"},
}


def _make_query(net):
    @torch.no_grad()
    def q(mu0, mu1, mu2, mu3, c, cv_sp, ln_sp):
        z = lambda v: torch.tensor([float(v)], device=DEVICE)
        Tc_ic = z(32.0)
        _,_,_,_,_,Tc = net(z(1.0), z(mu0), z(mu1), z(mu2), z(mu3),
                              z(c), z(cv_sp), z(ln_sp), Tc_ic)
        return float(Tc.item())
    return q


def _nmpc_query():
    oracle = CrystallizationNMPC()
    def q(mu0, mu1, mu2, mu3, c, cv_sp, ln_sp):
        return oracle.query(np.array([mu0, mu1, mu2, mu3, c]),
                              sp_CV=cv_sp, sp_Ln=ln_sp)
    return q


def _train_baseline(cfg, episodes, K1, K2, bs, label, seed=0):
    print(f"\n=== Train {label} (K={K1},{K2}, bs={bs}) ===")
    hp = CrystPINNHparams(K1=K1, K2=K2, bs=bs, **{k: float(cfg[k])
        for k in ["w_ode","w_ic","w_ytrk","w_utrk","w_du","w_u","w_x",
                   "lr1","lr2"]})
    net = PINN_Crystallization().to(DEVICE)
    t0 = time.time()
    hist = train_pinn_crystallization(net, episodes, hp, verbose=False, seed=seed)
    print(f"  trained in {time.time()-t0:.1f}s")
    return net, hist


def _scenario(cv_sp=1.0, ln_sp=15.0, x0=None):
    if x0 is None:
        op = CrystOperatingPoint()
        x0 = np.array([op.mu_0_0, op.mu_1_0, op.mu_2_0, op.mu_3_0, op.c_0])
    return {"x0": x0, "cv_sp": cv_sp, "ln_sp": ln_sp}


def _rollout(q, sc, params=None, n_steps=None):
    r = closed_loop_crystallization(q, sc["x0"], sc["cv_sp"], sc["ln_sp"],
                                       n_steps=n_steps, params=params)
    n = len(r.CV_trajectory)
    t = np.arange(1, n + 1) * CrystScenario().dt_hr
    return {"t": t, "CV": np.array(r.CV_trajectory),
             "Ln": np.array(r.Ln_trajectory),
             "Tc": np.array(r.Tc_trajectory),
             "c":  np.array(r.c_trajectory),
             "cv_sp": sc["cv_sp"], "ln_sp": sc["ln_sp"],
             "reward": r.cumulative_reward}


def _make_tracking_fig(rolls, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax_cv, ax_ln, ax_tc = axes
    any_r = next(iter(rolls.values()))

    ax_cv.axhline(any_r["cv_sp"], color="black", ls="--", lw=1.3,
                   label=r"$CV_{sp}$")
    for label, r in rolls.items():
        s = STYLES.get(label, {"color": "tab:blue", "lw": 2, "ls": "-"})
        ax_cv.plot(r["t"], r["CV"], color=s["color"], lw=s["lw"],
                    ls=s["ls"], label=f"{label}")
    ax_cv.set_ylabel("CV (coefficient of variation)")
    ax_cv.set_title("Crystallization closed-loop: CV tracking")
    ax_cv.legend(loc="best", fontsize=9)
    ax_cv.grid(alpha=0.3)

    ax_ln.axhline(any_r["ln_sp"], color="black", ls="--", lw=1.3,
                   label=r"$L_{n,sp}$")
    for label, r in rolls.items():
        s = STYLES.get(label, {"color": "tab:blue", "lw": 2, "ls": "-"})
        ax_ln.plot(r["t"], r["Ln"], color=s["color"], lw=s["lw"],
                    ls=s["ls"], label=f"{label}")
    ax_ln.set_ylabel(r"$L_n$ (number-mean length)")
    ax_ln.set_title(r"$L_n$ tracking")
    ax_ln.legend(loc="best", fontsize=9)
    ax_ln.grid(alpha=0.3)

    for label, r in rolls.items():
        s = STYLES.get(label, {"color": "tab:blue", "lw": 2, "ls": "-"})
        ax_tc.plot(r["t"], r["Tc"], color=s["color"], lw=s["lw"],
                    ls=s["ls"], label=f"{label}")
    ax_tc.set_ylabel(r"$T_c$ (deg C)")
    ax_tc.set_xlabel("Time (h)")
    ax_tc.set_title("Coolant temperature (control input)")
    ax_tc.axhline(T_C_LO, color="gray", ls=":", lw=0.8)
    ax_tc.axhline(T_C_HI, color="gray", ls=":", lw=0.8)
    ax_tc.legend(loc="best", fontsize=9)
    ax_tc.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-cfg",
                    default="results/pcgym_crystallization/crystallization/llm.json")
    ap.add_argument("--dpc-model",
                    default="results/two_phase_dpc_crystallization/pinn_cryst_p2_hard.pt")
    ap.add_argument("--llm-model",
                    default="results/two_phase_dpc_crystallization/pinn_cryst_llm_pretrain.pt")
    ap.add_argument("--output", default="results/paper_figures_crystallization/")
    ap.add_argument("--K1", type=int, default=10000)
    ap.add_argument("--K2", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n-train-eps", type=int, default=2000)
    ap.add_argument("--n-eval-reps", type=int, default=15,
                    help="Reps for evaluate_crystallization (NMPC slow!)")
    ap.add_argument("--skip-nmpc", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(a.best_cfg) as f:
        llm = json.load(f)
    best_cfg = llm["best_cfg"]
    print(f"LLM-best cfg: {best_cfg}\n")

    print(f"=== Sampling {a.n_train_eps} crystallization episodes... ===")
    t0 = time.time()
    episodes = sample_crystallization_episodes(N=a.n_train_eps, seed=a.seed,
                                                    query_nmpc=False, verbose=False)
    print(f"  done in {time.time()-t0:.1f}s\n")

    models = {}

    # 1. Bloor baseline
    if not a.skip_baseline:
        net_b, _ = _train_baseline(BLOOR_BASELINE_CFG, episodes,
                                       a.K1, a.K2, a.bs,
                                       "Bloor baseline PINN", seed=a.seed)
        torch.save(net_b.state_dict(), out_dir / "pinn_cryst_baseline.pt")
        models["Bloor PINN"] = net_b

    # 2. LLM-AutoOpt PINN
    if Path(a.llm_model).exists():
        print(f"\nLoading saved LLM PINN: {a.llm_model}")
        net_l = PINN_Crystallization().to(DEVICE)
        net_l.load_state_dict(torch.load(a.llm_model, map_location=DEVICE))
    else:
        net_l, _ = _train_baseline(best_cfg, episodes, a.K1, a.K2, a.bs,
                                       "LLM-AutoOpt PINN", seed=a.seed)
        torch.save(net_l.state_dict(), out_dir / "pinn_cryst_llm.pt")
    models["LLM PINN"] = net_l

    # 3. LLM+DPC PINN
    if Path(a.dpc_model).exists():
        print(f"\nLoading saved LLM+DPC PINN: {a.dpc_model}")
        net_d = PINN_Crystallization().to(DEVICE)
        net_d.load_state_dict(torch.load(a.dpc_model, map_location=DEVICE))
    else:
        print("\n=== Running two-phase DPC on LLM PINN ===")
        net_d = copy.deepcopy(net_l)
        cfg_easy = DPCRefineCfg(epochs=50, bs=16, lr=1e-5, mode="easy",
                                  seed=a.seed)
        refine_crystallization_dpc(net_d, cfg_easy, verbose=False,
                                       train_data=episodes)
        cfg_hard = DPCRefineCfg(epochs=150, bs=16, lr=5e-6, mode="hard",
                                  seed=a.seed + 1, early_stop_patience=30)
        refine_crystallization_dpc(net_d, cfg_hard, verbose=False,
                                       train_data=episodes)
        torch.save(net_d.state_dict(), out_dir / "pinn_cryst_dpc.pt")
    models["LLM+DPC PINN"] = net_d

    # ---- Tracking figure ----
    sc = _scenario(cv_sp=1.2, ln_sp=18.0)
    rolls = {}
    for label, net in models.items():
        print(f"[Rollout] {label} ...")
        rolls[label] = _rollout(_make_query(net), sc)
    if not a.skip_nmpc:
        print("[Rollout] NMPC oracle ... (slow)")
        rolls["NMPC oracle"] = _rollout(_nmpc_query(), sc)
    _make_tracking_fig(rolls, out_dir / "fig_cryst_tracking.png")

    # ---- Robustness (kinetic param mismatch) ----
    print("\n=== Robustness: k_a scaled by 0.7, 1.0, 1.3 ===")
    rolls_mm = {}
    p_default = CrystParams()
    for scale, label in [(0.7, "k_a x0.7"), (1.0, "k_a x1.0"),
                           (1.3, "k_a x1.3")]:
        p = CrystParams(
            k_a=p_default.k_a * scale, k_b=p_default.k_b, k_c=p_default.k_c,
            k_d=p_default.k_d, k_g=p_default.k_g, k_1=p_default.k_1,
            k_2=p_default.k_2, a=p_default.a, b=p_default.b,
            alpha=p_default.alpha, rho=p_default.rho)
        rolls_mm[label] = _rollout(_make_query(models["LLM+DPC PINN"]), sc,
                                       params=p)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_y, ax_u = axes
    ax_y.axhline(1.2, color="black", ls="--", lw=1.2, label=r"$CV_{sp}$")
    ax_y.axhline(18.0 / 15.0, color="gray", ls="--", lw=1.0,
                 label=r"$L_n/15$ for scaled view (sp=18)")
    cmap = {"k_a x0.7": "#1f77b4", "k_a x1.0": "#117A1F", "k_a x1.3": "#d62728"}
    for label, r in rolls_mm.items():
        ax_y.plot(r["t"], r["CV"], color=cmap[label], lw=1.8,
                   label=f"{label} CV(t)")
        ax_y.plot(r["t"], r["Ln"] / 15.0, color=cmap[label], lw=1.4,
                   ls="--", alpha=0.7, label=f"{label} L_n(t)/15")
        ax_u.plot(r["t"], r["Tc"], color=cmap[label], lw=1.6,
                   label=f"{label} Tc(t)")
    ax_y.set_ylabel("CV  /  L_n/15"); ax_y.grid(alpha=0.3)
    ax_y.set_title("Robustness under k_a mismatch (LLM+DPC PINN)")
    ax_y.legend(fontsize=8, ncol=2, loc="best")
    ax_u.set_ylabel(r"$T_c$ (deg C)"); ax_u.set_xlabel("Time (h)")
    ax_u.grid(alpha=0.3); ax_u.legend(fontsize=8, ncol=3, loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_cryst_robustness.png", dpi=150)
    plt.close(fig)
    print(f"  -> {out_dir/'fig_cryst_robustness.png'}")

    # ---- Setpoint sweep heatmap (3x3) for LLM+DPC PINN ----
    print("\n=== Setpoint sweep grid (LLM+DPC PINN, 9 SP combinations) ===")
    cv_grid = [0.7, 1.2, 1.8]
    ln_grid = [10.0, 15.0, 22.0]
    sweep_rolls = {}
    for cv in cv_grid:
        for ln in ln_grid:
            sweep_rolls[(cv, ln)] = _rollout(
                _make_query(models["LLM+DPC PINN"]),
                _scenario(cv_sp=cv, ln_sp=ln))
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    for i, cv in enumerate(cv_grid):
        for j, ln in enumerate(ln_grid):
            ax = axes[i][j]
            r = sweep_rolls[(cv, ln)]
            ax.plot(r["t"], r["CV"], color="#1F77B4", lw=1.5, label="CV(t)")
            ax.plot(r["t"], r["Ln"] / 15.0, color="#117A1F", lw=1.5,
                     ls="--", label=r"$L_n/15$")
            ax.axhline(cv, color="black", ls=":", lw=0.8)
            ax.axhline(ln / 15.0, color="gray", ls=":", lw=0.8)
            ax.set_title(f"CV_sp={cv}, L_n_sp={ln}", fontsize=9)
            ax.grid(alpha=0.3)
            if i == 2: ax.set_xlabel("Time (h)")
            if j == 0: ax.set_ylabel("Outputs")
            if i == 0 and j == 2: ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_cryst_setpoint_sweep.png", dpi=150)
    plt.close(fig)
    print(f"  -> {out_dir/'fig_cryst_setpoint_sweep.png'}")

    # ---- Metrics table ----
    print(f"\n=== Computing metrics ({a.n_eval_reps} reps each) ===")
    metrics_log = {}
    for label, net in models.items():
        print(f"  evaluating {label}...")
        metrics_log[label] = evaluate_crystallization(
            _make_query(net), n_reps=a.n_eval_reps, seed=42, verbose=False)
    with (out_dir / "table_cryst_metrics.csv").open("w") as f:
        f.write("controller,median_reward,optimality_gap,MAD\n")
        for label, m in metrics_log.items():
            f.write(f"{label},{m['median_reward_pi']:.4f},"
                    f"{m['optimality_gap']:.4f},{m['MAD']:.4f}\n")
    print(f"  -> {out_dir/'table_cryst_metrics.csv'}")

    print("\n=== Final metrics ===")
    print(f"{'controller':<25}{'median R':>15}{'opt gap':>12}{'MAD':>10}")
    for label, m in metrics_log.items():
        print(f"  {label:<25}{m['median_reward_pi']:>15.4f}"
              f"{m['optimality_gap']:>12.4f}{m['MAD']:>10.4f}")

    summary = {
        "case": "crystallization",
        "llm_best_cfg": best_cfg,
        "metrics": {label: {k: float(m[k]) for k in
                              ("median_reward_pi", "median_reward_oracle",
                               "optimality_gap", "MAD")}
                       for label, m in metrics_log.items()},
    }
    with (out_dir / "summary_cryst.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nAll outputs saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
