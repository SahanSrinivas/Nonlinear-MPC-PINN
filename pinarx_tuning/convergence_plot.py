"""LLMAgentOpt vs Optuna-TPE convergence comparison.

For each pair of studies (one LLM-driven, one pure TPE), plot the
cumulative-best objective vs trial number. A 3-panel figure
(noiseless / SNR=100 / SNR=35) makes the headline point of the paper:
the LLM's win comes from the noisy regime.

Usage:
  python convergence_plot.py \
      --pair noiseless runs/noiseless_lean runs/noiseless_tpe \
      --pair snr100    runs/snr100_lean    runs/snr100_tpe \
      --pair snr35     runs/snr35_lean     runs/snr35_tpe \
      --out runs/figs/convergence_3panel.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_objectives(study_dir: str) -> np.ndarray:
    """Pull the `objective` value out of every trial in results.json."""
    path = Path(study_dir) / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"No results.json at {path}")
    rows = json.loads(path.read_text())
    objs = []
    for r in rows:
        v = r.get("objective", r.get("val_loss"))
        if v is None or not np.isfinite(float(v)):
            continue
        objs.append(float(v))
    if not objs:
        raise ValueError(f"No usable objectives in {path}")
    return np.array(objs)


def cumulative_best(objs: np.ndarray) -> np.ndarray:
    out = np.empty_like(objs)
    out[0] = objs[0]
    for i in range(1, len(objs)):
        out[i] = min(out[i - 1], objs[i])
    return out


def plot_panel(ax, llm_dir: str, tpe_dir: str, title: str):
    llm_objs = load_objectives(llm_dir)
    tpe_objs = load_objectives(tpe_dir)
    llm_best = cumulative_best(llm_objs)
    tpe_best = cumulative_best(tpe_objs)

    x_llm = np.arange(1, len(llm_best) + 1)
    x_tpe = np.arange(1, len(tpe_best) + 1)

    ax.plot(x_llm, llm_best, "-", lw=2.0, color="#1f77b4",
              label=f"LLMAgentOpt (best={llm_best[-1]:.3e})")
    ax.plot(x_tpe, tpe_best, "--", lw=2.0, color="#d62728",
              label=f"Optuna TPE  (best={tpe_best[-1]:.3e})")
    ax.scatter(x_llm, llm_objs, s=10, c="#1f77b4", alpha=0.35,
                  label="_nolegend_")
    ax.scatter(x_tpe, tpe_objs, s=10, c="#d62728", alpha=0.35,
                  label="_nolegend_")
    ax.set_yscale("log")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Cumulative-best objective\n(0.5 * T1 + 0.5 * T2  MAE)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", action="append", nargs=3,
                    metavar=("LABEL", "LLM_DIR", "TPE_DIR"),
                    help="Add a (label, llm_study, tpe_study) panel. "
                         "May be passed multiple times.")
    ap.add_argument("--out", default="runs/figs/convergence_3panel.png")
    args = ap.parse_args()
    if not args.pair:
        ap.error("at least one --pair required")

    n = len(args.pair)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
    for ax, (label, llm_dir, tpe_dir) in zip(axes[0], args.pair):
        plot_panel(ax, llm_dir, tpe_dir, label)
    fig.tight_layout()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
