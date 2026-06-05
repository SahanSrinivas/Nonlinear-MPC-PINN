"""Render the paper-comparison table for Res-Phys NARX.

Reads a JSON result file produced by train.py (or a list of them via the
LEAN tuner) and prints a comparison row matching the paper's Table 2 layout:

  Model         | Test 1 (Within range) | Test 2 (Extrapolation)
  --------------|-----------------------|------------------------
  NARX (paper)  | 0.001508              | 0.01934
  PI-NARX (paper)| 0.001242             | 0.01556
  Res-Phys NARX | <ours>                | <ours>

Designed for: copy/paste into a LaTeX table, or into the paper's results
section. Treats the paper's NARX/PI-NARX numbers as published references
to compare against (we do NOT try to reproduce them; see README).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PAPER_TABLE_2 = {
    "NARX":     {"t1": 0.001508, "t2": 0.01934},
    "PI-NARX":  {"t1": 0.001242, "t2": 0.01556},
}

PAPER_TABLE_6 = {
    "SNR 35": {"NARX":    {"t1": 0.009947, "t2": 0.03369},
                "PI-NARX": {"t1": 0.009269, "t2": 0.02937}},
    "SNR 100":{"NARX":    {"t1": 0.009771, "t2": 0.03321},
                "PI-NARX": {"t1": 0.009049, "t2": 0.02916}},
}


def fmt(x: float | None) -> str:
    return f"{x:.6f}" if x is not None else "-"


def render_table(results: list[dict],
                   reference: dict = PAPER_TABLE_2,
                   noise: str | None = None) -> str:
    """Render the comparison table. `results` is a list of dicts from
    train.run_trial - one per Res-Phys NARX configuration to compare."""
    lines = []
    if noise is None:
        lines.append("Comparison vs paper Table 2 (noiseless)")
    else:
        lines.append(f"Comparison vs paper Table 6 ({noise})")
    lines.append("=" * 74)
    lines.append(f"  {'Model':<28}{'Test 1 (Within range)':>22}"
                   f"{'Test 2 (Extrapolation)':>22}")
    lines.append("  " + "-" * 72)
    for name, m in reference.items():
        lines.append(f"  {name + ' (paper)':<28}"
                       f"{fmt(m['t1']):>22}{fmt(m['t2']):>22}")
    lines.append("  " + "-" * 72)
    for r in results:
        label = r.get("label",
                       f"Res-Phys NARX (seed {r['hp']['seed']})")
        t1 = r.get("mae_t1_one_step")
        t2 = r.get("mae_t2_one_step")
        lines.append(f"  {label:<28}{fmt(t1):>22}{fmt(t2):>22}")
    # vs-paper line for the BEST Res-Phys result
    if results:
        best = min(results, key=lambda r: r["objective"])
        bt1, bt2 = best["mae_t1_one_step"], best["mae_t2_one_step"]
        ratio_narx_t1   = PAPER_TABLE_2["NARX"]["t1"] / bt1
        ratio_narx_t2   = PAPER_TABLE_2["NARX"]["t2"] / bt2
        ratio_pinarx_t1 = PAPER_TABLE_2["PI-NARX"]["t1"] / bt1
        ratio_pinarx_t2 = PAPER_TABLE_2["PI-NARX"]["t2"] / bt2
        lines.append("")
        lines.append(f"  Best Res-Phys NARX vs paper:")
        lines.append(f"    vs paper NARX     : Test1 {ratio_narx_t1:.1f}x better, "
                       f"Test2 {ratio_narx_t2:.1f}x better")
        lines.append(f"    vs paper PI-NARX  : Test1 {ratio_pinarx_t1:.1f}x better, "
                       f"Test2 {ratio_pinarx_t2:.1f}x better")
    return "\n".join(lines)


def render_latex(results: list[dict]) -> str:
    """LaTeX booktabs version - drop straight into the paper."""
    out = ["\\begin{tabular}{lrr}",
            "\\toprule",
            "Model & Test 1 (Within range) & Test 2 (Extrapolation) \\\\",
            "\\midrule"]
    for name, m in PAPER_TABLE_2.items():
        out.append(f"{name} (paper) & {m['t1']:.6f} & {m['t2']:.6f} \\\\")
    out.append("\\midrule")
    for r in results:
        label = r.get("label",
                       f"Res-Phys NARX (seed {r['hp']['seed']})")
        out.append(f"{label} & {r['mae_t1_one_step']:.6f} & "
                     f"{r['mae_t2_one_step']:.6f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def load_results(paths: list[str]) -> list[dict]:
    results = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            for f in sorted(p.glob("*.json")):
                results.append(json.loads(f.read_text()))
        else:
            results.append(json.loads(p.read_text()))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+",
                    help="JSON file(s) or directory of results from train.py")
    ap.add_argument("--latex", action="store_true",
                    help="Also print LaTeX booktabs table")
    ap.add_argument("--noise", default=None,
                    help="Mark results as 'snr35' or 'snr100' for Table 6")
    args = ap.parse_args()

    results = load_results(args.results)
    if args.noise == "snr35":
        ref = {k: v["SNR 35"] for k, v in PAPER_TABLE_6.items()}
        # Reshape to match render_table signature
        ref = {f"{m} SNR 35": PAPER_TABLE_6["SNR 35"][m]
                for m in ("NARX", "PI-NARX")}
        print(render_table(results, reference=ref, noise="SNR 35"))
    elif args.noise == "snr100":
        ref = {f"{m} SNR 100": PAPER_TABLE_6["SNR 100"][m]
                for m in ("NARX", "PI-NARX")}
        print(render_table(results, reference=ref, noise="SNR 100"))
    else:
        print(render_table(results))
    if args.latex:
        print()
        print(render_latex(results))
