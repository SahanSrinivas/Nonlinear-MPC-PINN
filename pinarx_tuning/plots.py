"""Reproduce the paper's Fig 3 / 7 / 8 / A.2 / A.3 style plots for our
Res-Phys NARX model.

All plots use the same visual conventions as Thosar et al. 2025:
  - 2x2 grid of subplots for [C_A, T, T_c, h]
  - True Value : black solid
  - Res-Phys NARX : green dashed (paper used green for PI-NARX)
  - Optional NARX baseline : red dashed (we draw it if the user passes one)
  - x-axis: Time (min)
  - y-axes match paper's per-variable units (mol/L, K, K, m)

Public functions:
  plot_test_case_4panel(model, traj, save_path, title="")
  plot_test_case_overlay(models, traj, save_path, title="")  # multiple models
  plot_training_data(train_traj, val_traj, save_path)        # Fig A.2
  plot_validation_fit(model, val_traj, save_path)            # Fig A.3
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")     # headless / Colab-safe
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Style constants - mirror the paper
# ============================================================================
CHANNEL_NAMES  = ["C_A", "T", "T_c", "h"]
CHANNEL_LABELS = [r"$C_A$ (mol/L)", r"$T$ (K)", r"$T_c$ (K)", r"$h$ (m)"]

TRUE_KW   = dict(color="black", linestyle="-",  linewidth=1.4, label="True value")
RESPHYS_KW = dict(color="green", linestyle="--", linewidth=1.4, label="Res-Phys NARX")
NARX_KW    = dict(color="red",   linestyle="--", linewidth=1.0, label="NARX (paper, reference)")


# ============================================================================
# Helper: roll out a model on a trajectory's inputs starting from its true
# initial window.
# ============================================================================
def _model_predict_traj(model, traj: dict) -> np.ndarray:
    """Return the model's autoregressive prediction of `traj` (N+1, n_y)."""
    w = model.hp.window
    y_pred = model.rollout(traj["u"], y_init=traj["y"][:w])
    return y_pred


# ============================================================================
# Fig 3 / 7 / 8 - one test case, 4-panel comparison
# ============================================================================
def plot_test_case_4panel(model, traj: dict, save_path: str,
                            title: str = "",
                            include_one_step: bool = False) -> str:
    """Plot True Value vs Res-Phys NARX (autoregressive) on a 2x2 grid.

    If `include_one_step=True`, also overlay the model's one-step-teacher-forced
    prediction (lighter green dotted) - useful to see autoregressive drift.

    Saves PNG to `save_path` and returns the path.
    """
    y_true = traj["y"]
    y_pred = _model_predict_traj(model, traj)
    N = y_true.shape[0]
    t = np.arange(N)

    if include_one_step:
        # Teacher-forced: feed true window at each step
        from nn_utils import make_windows
        X, _ = make_windows(traj["u"], y_true, model.hp.window,
                              include_u_lags=model.hp.include_u_lags)
        # Build one-step prediction array aligned to time
        y_one = np.zeros_like(y_pred)
        y_one[:model.hp.window] = y_true[:model.hp.window]
        import torch
        device = model.hp.device
        u_start = model.hp.window * model.n_y
        y_prev = torch.from_numpy(X[:, :model.n_y].astype(np.float32)).to(device)
        u_t    = torch.from_numpy(X[:, u_start:u_start + model.n_u
                                     ].astype(np.float32)).to(device)
        X_t = torch.from_numpy(model.x_norm.transform(X)).to(device)
        model.net.eval()
        with torch.no_grad():
            # For Res-Phys: forward_norm uses cached scipy step
            Y_n = model._forward_norm(X_t, y_prev, u_t).cpu().numpy()
        model.net.train()
        y_one[model.hp.window:] = model.y_norm.inverse(Y_n)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, name, label in zip(axes.ravel(), CHANNEL_NAMES, CHANNEL_LABELS):
        c = CHANNEL_NAMES.index(name)
        ax.plot(t, y_true[:, c], **TRUE_KW)
        ax.plot(t, y_pred[:, c], **RESPHYS_KW)
        if include_one_step:
            ax.plot(t, y_one[:, c], color="green", linestyle=":",
                      linewidth=0.9, alpha=0.7, label="Res-Phys (one-step)")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    # Single legend on the top-left axis
    axes[0, 0].legend(loc="best", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Test case 1 vs 2 side-by-side (Fig 3 in paper)
# ============================================================================
def plot_fig3_style(model, traj_t1: dict, traj_t2: dict,
                      save_path: str) -> str:
    """Reproduce paper Fig 3: two 4-panel plots stacked (a) Test 1, (b) Test 2."""
    fig = plt.figure(figsize=(11, 13))
    for row_idx, (traj, label) in enumerate([(traj_t1, "Test case 1 - interpolation"),
                                                (traj_t2, "Test case 2 - extrapolation")]):
        y_true = traj["y"]
        y_pred = _model_predict_traj(model, traj)
        t = np.arange(y_true.shape[0])
        for col_idx, (name, ylabel) in enumerate(zip(CHANNEL_NAMES, CHANNEL_LABELS)):
            ax = fig.add_subplot(4, 2, row_idx * 4 + col_idx + 1)
            # Reorganize so that (row=0..3 is channel, col=0..1 is test case)
            # Actually easier: put each test case in its own 2x2 block
        # the loop above doesn't produce the right layout; rewriting below
    plt.close(fig)

    # Cleaner: two separate figures stacked via subplot mosaic
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex="col")
    for col_idx, (traj, label) in enumerate(
            [(traj_t1, "(a) Test case 1 - interpolation"),
             (traj_t2, "(b) Test case 2 - extrapolation")]):
        y_true = traj["y"]
        y_pred = _model_predict_traj(model, traj)
        t = np.arange(y_true.shape[0])
        for row_idx, (name, ylabel) in enumerate(zip(CHANNEL_NAMES, CHANNEL_LABELS)):
            ax = axes[row_idx, col_idx]
            ax.plot(t, y_true[:, row_idx], **TRUE_KW)
            ax.plot(t, y_pred[:, row_idx], **RESPHYS_KW)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if row_idx == 0:
                ax.set_title(label, fontsize=10)
                ax.legend(loc="best", fontsize=8)
            if row_idx == 3:
                ax.set_xlabel("Time (min)")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Fig A.2 - training data outputs + input profile
# ============================================================================
def plot_training_data(train_traj: dict, val_traj: dict,
                         save_path: str) -> str:
    """Reproduce paper Fig A.2: (a) training+validation OUTPUT trajectories,
    (b) training+validation INPUT (Q_f, Q_c) profile."""
    # Stitch train and val (they're contiguous in our protocol)
    y_full = np.concatenate([train_traj["y"][:-1], val_traj["y"]], axis=0)
    u_full = np.concatenate([train_traj["u"], val_traj["u"]], axis=0)
    n_train = train_traj["u"].shape[0]
    N = u_full.shape[0]
    t_y = np.arange(y_full.shape[0])
    t_u = np.arange(N)

    fig = plt.figure(figsize=(11, 9))

    # (a) outputs - 2x2 grid
    for c, (name, ylabel) in enumerate(zip(CHANNEL_NAMES, CHANNEL_LABELS)):
        ax = fig.add_subplot(3, 2, c + 1)
        ax.plot(t_y[:n_train + 1], y_full[:n_train + 1, c],
                  color="black", linestyle="-", linewidth=1.0, label="Training")
        ax.plot(t_y[n_train:], y_full[n_train:, c],
                  color="black", linestyle="--", linewidth=1.0, label="Validation")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if c == 0:
            ax.legend(loc="best", fontsize=8)

    # (b) inputs - one wide subplot underneath (rows 5-6 of the 3x2 grid)
    ax = fig.add_subplot(3, 1, 3)
    ax.plot(t_u[:n_train], u_full[:n_train, 0], color="black", linestyle="-",
              linewidth=1.0, label=r"$Q_f$ (train)")
    ax.plot(t_u[n_train:], u_full[n_train:, 0], color="black", linestyle="--",
              linewidth=1.0, label=r"$Q_f$ (val)")
    ax.plot(t_u[:n_train], u_full[:n_train, 1], color="red", linestyle="-",
              linewidth=1.0, label=r"$Q_c$ (train)")
    ax.plot(t_u[n_train:], u_full[n_train:, 1], color="red", linestyle="--",
              linewidth=1.0, label=r"$Q_c$ (val)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Input flowrates (L/min)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axvline(n_train, color="gray", linestyle=":", linewidth=0.8)

    fig.suptitle("Training and validation data (paper Fig A.2 style)",
                   fontsize=11)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Fig A.3 - validation data performance
# ============================================================================
def plot_validation_fit(model, val_traj: dict, save_path: str) -> str:
    """Reproduce paper Fig A.3: model prediction vs true value on the
    VALIDATION trajectory, 2x2 grid of outputs."""
    return plot_test_case_4panel(model, val_traj, save_path,
                                     title="Validation data fit (paper Fig A.3 style)")


# ============================================================================
# Fig 5 - Limited-data ablation, C_A only (2x2 grid, one panel per data size)
# ============================================================================
def plot_fig5_style(models_by_size: dict, test1: dict, test2: dict,
                      save_path_a: str, save_path_b: str) -> tuple[str, str]:
    """Reproduce paper Fig 5: 2x2 grid of C_A plots, one per training data
    size, showing True Value (black) vs our Res-Phys NARX (green dashed).
    Optionally include a NARX-reference dashed red line if `models_by_size`
    values are dicts with both 'resphys' and 'narx' keys.

    models_by_size: dict mapping data-size (int) -> trained ResPhysNARXModel
                     OR dict like {'resphys': model, 'narx': narx_model}

    Returns (path_for_test1, path_for_test2).
    """
    def _plot_one(test_traj, label, save_path):
        sizes = sorted(models_by_size.keys())
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        for ax, n in zip(axes.ravel(), sizes):
            entry = models_by_size[n]
            if isinstance(entry, dict):
                m_rp   = entry.get("resphys")
                m_narx = entry.get("narx")
            else:
                m_rp, m_narx = entry, None
            y_true = test_traj["y"]
            t = np.arange(y_true.shape[0])
            # C_A only (channel 0)
            ax.plot(t, y_true[:, 0], **TRUE_KW,
                      ) if False else ax.plot(t, y_true[:, 0],
                                                  color="black", linestyle="-",
                                                  linewidth=1.4,
                                                  label=f"{n} points: True Value")
            if m_narx is not None:
                y_narx = _model_predict_traj(m_narx, test_traj)
                ax.plot(t, y_narx[:, 0], color="red", linestyle="--",
                          linewidth=1.0, label="NARX")
            y_rp = _model_predict_traj(m_rp, test_traj)
            ax.plot(t, y_rp[:, 0], color="green", linestyle="--",
                      linewidth=1.4, label="Res-Phys NARX")
            ax.set_ylabel(r"$C_A$ (mol/L)")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[1, 0].set_xlabel("Time (min)")
        axes[1, 1].set_xlabel("Time (min)")
        fig.suptitle(f"({label}) Test case {label[-1]} - "
                       f"{'interpolation' if label.endswith('a') else 'extrapolation'}: "
                       f"$C_A$ at different training sizes", fontsize=11)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return save_path

    p1 = _plot_one(test1, "fig5a", save_path_a)
    p2 = _plot_one(test2, "fig5b", save_path_b)
    return p1, p2


# ============================================================================
# Fig 6 - Limited-data ablation, ALL 4 outputs in one figure
# ============================================================================
def plot_fig6_style(models_by_size: dict, test_traj: dict,
                      save_path: str, test_label: str = "Test case") -> str:
    """Reproduce paper Fig 6: 2x2 grid of channels, multiple data-size
    predictions overlaid as different colored dashed lines.

    Each Res-Phys NARX model trained on a different data size gets its own
    color; the true value is always black solid.
    """
    sizes = sorted(models_by_size.keys())
    # Colors mimicking paper Fig 6: small data = dashed in distinct colors
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728"]   # blue, green, purple, red
    color_for = {n: palette[i % len(palette)] for i, n in enumerate(sizes)}

    y_true = test_traj["y"]
    t = np.arange(y_true.shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, name, ylabel in zip(axes.ravel(), CHANNEL_NAMES, CHANNEL_LABELS):
        c = CHANNEL_NAMES.index(name)
        ax.plot(t, y_true[:, c], color="black", linestyle="-",
                  linewidth=1.4, label="True value")
        for n in sizes:
            entry = models_by_size[n]
            m_rp = entry["resphys"] if isinstance(entry, dict) else entry
            y_rp = _model_predict_traj(m_rp, test_traj)
            ax.plot(t, y_rp[:, c], color=color_for[n], linestyle="--",
                      linewidth=1.2, label=f"{n} points")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[1, 0].set_xlabel("Time (min)")
    axes[1, 1].set_xlabel("Time (min)")
    fig.suptitle(f"{test_label}: Res-Phys NARX with varying training sizes",
                   fontsize=11)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Driver - call after training to dump all the paper-style figures
# ============================================================================
def dump_paper_figs(model, train_traj: dict, val_traj: dict,
                      test1: dict, test2: dict,
                      out_dir: str,
                      noise_label: str = "noiseless") -> dict:
    """Save all the standard figures for one trained model. Returns dict
    of {fig_name -> path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["test1"]    = plot_test_case_4panel(model, test1,
                          str(out_dir / f"fig3a_test1_{noise_label}.png"),
                          title=f"Test case 1 - interpolation  ({noise_label})")
    paths["test2"]    = plot_test_case_4panel(model, test2,
                          str(out_dir / f"fig3b_test2_{noise_label}.png"),
                          title=f"Test case 2 - extrapolation  ({noise_label})")
    paths["fig3"]     = plot_fig3_style(model, test1, test2,
                          str(out_dir / f"fig3_combined_{noise_label}.png"))
    paths["training"] = plot_training_data(train_traj, val_traj,
                          str(out_dir / f"figA2_training_data_{noise_label}.png"))
    paths["val_fit"]  = plot_validation_fit(model,  val_traj,
                          str(out_dir / f"figA3_val_fit_{noise_label}.png"))
    return paths


# ============================================================================
# CLI - takes a saved trial JSON + checkpoint, dumps figures
# (used by Colab once a model is trained)
# ============================================================================
if __name__ == "__main__":
    import argparse
    import torch
    from resphys_narx import ResPhysNARXModel, ResPhysNARXHparams
    from data_gen import (gen_grid_train_val_split, gen_test1_set, gen_test2_set,
                            add_noise, SNR_T6_NOISY_LO, SNR_T6_NOISY_HI)
    from train import build_hparams

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--noise",   choices=[None, "snr35", "snr100"], default=None)
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available()
                                            else "cpu")
    ap.add_argument("--epochs",  type=int, default=1000)
    ap.add_argument("--lbfgs",   type=int, default=1000)
    ap.add_argument("--out-dir", default="runs/figs")
    args = ap.parse_args()

    print(f"Training Res-Phys NARX  (noise={args.noise}, device={args.device})...")
    tr, va = gen_grid_train_val_split(qf_levels=10, qc_levels=10, seed=args.seed)
    t1, t2 = gen_test1_set(), gen_test2_set()
    if args.noise is not None:
        snr_vec = SNR_T6_NOISY_LO if args.noise == "snr35" else SNR_T6_NOISY_HI
        tr = {**tr, "y": add_noise(tr["y"], snr_vec, seed=42)}
        va = {**va, "y": add_noise(va["y"], snr_vec, seed=43)}
        t1 = {**t1, "y": add_noise(t1["y"], snr_vec, seed=44)}
        t2 = {**t2, "y": add_noise(t2["y"], snr_vec, seed=45)}
    hp = build_hparams(dict(seed=args.seed, device=args.device,
                              n_epochs_adam=args.epochs, lbfgs_iters=args.lbfgs))
    model = ResPhysNARXModel(n_u=2, n_y=4, hp=hp)
    model.fit(tr, va, verbose=True)

    noise_label = args.noise if args.noise else "noiseless"
    paths = dump_paper_figs(model, tr, va, t1, t2, args.out_dir, noise_label)
    print("\nSaved figures:")
    for name, p in paths.items():
        print(f"  {name:<10} -> {p}")
