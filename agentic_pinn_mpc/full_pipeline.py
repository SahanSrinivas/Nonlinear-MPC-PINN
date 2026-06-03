"""End-to-end pipeline: hparam tuning (Phase A) -> base train -> DPC refinement
(Phase C) -> closed-loop evaluation -> compare against Kardamaki baseline.

Optionally also runs:
  - Phase B (agentic training monitor) during the base-train step
  - Architecture ablation (Kardamaki vs SIREN vs ResNet vs wider/deeper)

This is the deployment-ready entry-point for the full Paper 3 RunPod run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .pinn_siso import DEVICE, PINNHparams, train_pinn_siso
from .evaluate import evaluate_model
from .tuners import KARDAMAKI_BEST, make_tuner
from .rl_refine import refine_with_dpc, DPCRefineCfg
from .advanced_arch import ARCH_REGISTRY, count_params
from .bench import load_training_data


def base_train(hparams: dict, x0_all, u0_all, ysp_all, d0_all,
                K1: int, K2: int, bs: int, arch_name: str = "kardamaki_64_16_16",
                seed: int = 0, verbose: bool = False):
    """Train a PINN from scratch with the given architecture + hparams.

    Returns (model, training_history).
    """
    hp = PINNHparams(
        w_ode=hparams["w_ode"], w_ic=hparams["w_ic"],
        w_ytrk=hparams["w_ytrk"], w_utrk=hparams["w_utrk"],
        w_du=hparams["w_du"], w_u=hparams["w_u"], w_x=hparams["w_x"],
        lr1=hparams["lr1"], lr2=hparams["lr2"],
        K1=K1, K2=K2, bs=bs,
    )
    if arch_name == "kardamaki_64_16_16":
        return train_pinn_siso(hp, x0_all, u0_all, ysp_all, d0_all,
                                 verbose=verbose, seed=seed)
    # Custom architecture: build it, then run the same training loop.
    # We monkey-patch the model into the training function by reusing
    # train_pinn_siso's structure but with our custom model factory.
    from .pinn_siso import (PINN_Controller, generate_collocation_points)
    torch.manual_seed(seed)
    model = ARCH_REGISTRY[arch_name]()
    t_wp = torch.cat([torch.arange(0, hp.T_horizon, hp.Ts, device=DEVICE),
                       torch.tensor([hp.T_horizon], device=DEVICE)])
    N_WP = t_wp.shape[0]
    t_col = generate_collocation_points(
        horizon=hp.T_horizon, dt_dense=0.01, dt_medium=0.1, dt_sparse=1.0)
    N_COL = t_col.shape[0]
    hist_p1, hist_p2 = [], []
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr1)
    for ep in range(1, hp.K1 + 1):
        s, e = (ep - 1) * hp.bs, ep * hp.bs
        loss, _ = model.loss(hp.bs, t_wp, N_WP, t_col, N_COL,
                              x0_all[s:e], u0_all[s:e],
                              ysp_all[s:e], d0_all[s:e],
                              hp.w_ode, hp.w_ic, hp.w_ytrk, hp.w_utrk,
                              0.0, 0.0, 0.0)
        if torch.isnan(loss):
            return model, {"hist_p1": hist_p1, "hist_p2": [],
                            "nan_at": ("P1", ep)}
        opt.zero_grad(); loss.backward(); opt.step()
        hist_p1.append(loss.item())
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr2)
    for ep in range(1, hp.K2 + 1):
        s, e = (ep - 1) * hp.bs, ep * hp.bs
        loss, _ = model.loss(hp.bs, t_wp, N_WP, t_col, N_COL,
                              x0_all[s:e], u0_all[s:e],
                              ysp_all[s:e], d0_all[s:e],
                              hp.w_ode, hp.w_ic, hp.w_ytrk, hp.w_utrk,
                              hp.w_du, hp.w_u, hp.w_x)
        if torch.isnan(loss):
            return model, {"hist_p1": hist_p1, "hist_p2": hist_p2,
                            "nan_at": ("P2", ep)}
        opt.zero_grad(); loss.backward(); opt.step()
        hist_p2.append(loss.item())
    return model, {"hist_p1": hist_p1, "hist_p2": hist_p2, "nan_at": None}


def run_pipeline(hparams: dict, x0_all, u0_all, ysp_all, d0_all,
                  arch_name: str, K1: int, K2: int, bs: int,
                  refine: bool = True, refine_epochs: int = 200,
                  refine_lr: float = 1e-5,
                  n_eval_tracking: int = 500, n_eval_disturbance: int = 500,
                  seed: int = 0) -> dict:
    """One end-to-end pipeline run. Returns metrics dict."""
    timing = {}
    t0 = time.time()
    model, _ = base_train(hparams, x0_all, u0_all, ysp_all, d0_all,
                            K1, K2, bs, arch_name=arch_name, seed=seed)
    timing["base_train_s"] = time.time() - t0

    t0 = time.time()
    m_before = evaluate_model(model, n_tracking=n_eval_tracking,
                                n_disturbance=n_eval_disturbance, seed=seed + 42)
    timing["eval_before_s"] = time.time() - t0

    m_after = None
    if refine:
        t0 = time.time()
        cfg = DPCRefineCfg(epochs=refine_epochs, lr=refine_lr,
                            seed=seed)
        model, _ = refine_with_dpc(model, cfg)
        timing["refine_s"] = time.time() - t0
        t0 = time.time()
        m_after = evaluate_model(model, n_tracking=n_eval_tracking,
                                   n_disturbance=n_eval_disturbance,
                                   seed=seed + 42)
        timing["eval_after_s"] = time.time() - t0

    return {
        "arch": arch_name,
        "arch_params": count_params(model),
        "metrics_before_refine": m_before,
        "metrics_after_refine": m_after,
        "timing": timing,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["kardamaki_64_16_16"],
                     choices=list(ARCH_REGISTRY.keys()),
                     help="Architectures to test")
    ap.add_argument("--K1", type=int, default=500)
    ap.add_argument("--K2", type=int, default=500)
    ap.add_argument("--bs", type=int, default=50)
    ap.add_argument("--refine-epochs", type=int, default=100)
    ap.add_argument("--refine-lr", type=float, default=1e-5)
    ap.add_argument("--n-eval-tracking", type=int, default=200)
    ap.add_argument("--n-eval-disturbance", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default="results/full_pipeline_v1")
    a = ap.parse_args()

    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    print(f"=== Paper 3 full pipeline ===")
    print(f"  Device: {DEVICE}")
    print(f"  Architectures: {a.archs}")
    print(f"  Base train: K1={a.K1}, K2={a.K2}, bs={a.bs}")
    print(f"  DPC refine: {a.refine_epochs} epochs, lr={a.refine_lr}")
    print(f"  Eval: {a.n_eval_tracking}+{a.n_eval_disturbance} episodes")
    print()
    print("Loading training data...")
    x0_all, u0_all, ysp_all, d0_all = load_training_data()
    print(f"  Loaded {x0_all.shape[0]} episodes")

    # Use Kardamaki's published-best hparams (no LLM tuning here - that's bench.py)
    hparams = dict(KARDAMAKI_BEST)
    results = []
    for arch_name in a.archs:
        print(f"\n=== Architecture: {arch_name} ===")
        r = run_pipeline(hparams, x0_all, u0_all, ysp_all, d0_all,
                          arch_name=arch_name,
                          K1=a.K1, K2=a.K2, bs=a.bs,
                          refine=True,
                          refine_epochs=a.refine_epochs,
                          refine_lr=a.refine_lr,
                          n_eval_tracking=a.n_eval_tracking,
                          n_eval_disturbance=a.n_eval_disturbance,
                          seed=a.seed)
        results.append(r)
        m_b = r["metrics_before_refine"]
        m_a = r["metrics_after_refine"]
        print(f"  Params: {r['arch_params']}")
        print(f"  Base   eval: tracking_mean={m_b['tracking_mean_offset_m']:.4f}, "
              f"disturbance_mean={m_b['disturbance_mean_offset_m']:.4f}")
        if m_a is not None:
            print(f"  Refined eval: tracking_mean={m_a['tracking_mean_offset_m']:.4f}, "
                  f"disturbance_mean={m_a['disturbance_mean_offset_m']:.4f}")
            delta_t = (m_a['tracking_mean_offset_m']
                        - m_b['tracking_mean_offset_m'])
            delta_d = (m_a['disturbance_mean_offset_m']
                        - m_b['disturbance_mean_offset_m'])
            print(f"  Delta (refine): tracking {delta_t:+.4f}, "
                  f"disturbance {delta_d:+.4f}")
        print(f"  Timing: {r['timing']}")
    with (out / "results.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out.resolve()}")


if __name__ == "__main__":
    main()
