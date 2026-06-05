"""Pull LLM rationales from an LLMAgentOpt run and emit a paper-ready
markdown listing (Appendix or sidebar figure).

For each trial that has `llm_rationale != None`, prints:
  - Trial # / mode / objective / Test1 / Test2
  - The Strategy agent's full natural-language rationale
  - The Tuning agent's JSON config delta (parsed from `hp`)

Usage:
  python extract_rationales.py runs/snr100_lean \
      --top 3 \
      --out runs/snr100_lean/rationales.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study_dir")
    ap.add_argument("--top", type=int, default=3,
                    help="Emit only the N lowest-objective LLM-driven trials "
                         "(plus the very first warm-start). 0 = all.")
    ap.add_argument("--out", default=None,
                    help="Write markdown here. Default: <study>/rationales.md")
    args = ap.parse_args()

    sd = Path(args.study_dir)
    rows = json.loads((sd / "results.json").read_text())
    llm_rows = [r for r in rows if r.get("llm_rationale")]
    if not llm_rows:
        print(f"No llm_rationale entries in {sd}/results.json -- was this a "
                "lean3 / llm_agent_opt run?")
        return

    llm_rows_sorted = sorted(llm_rows, key=lambda r: r.get("objective",
                                                                float("inf")))
    if args.top > 0:
        picked_ids = {id(llm_rows[0])}                    # always include first
        picked_ids.update(id(r) for r in llm_rows_sorted[:args.top])
        chosen = [r for r in llm_rows if id(r) in picked_ids]
    else:
        chosen = llm_rows

    lines = [f"# LLMAgentOpt rationale trace -- {sd.name}",
              "",
              f"_{len(llm_rows)} LLM-driven trials out of {len(rows)} total "
              f"(remainder were pure-TPE picks). Showing {len(chosen)}._",
              ""]
    for i, r in enumerate(chosen):
        trial_idx = rows.index(r)
        hp = {k: v for k, v in (r.get("hp") or {}).items()
                if k not in ("device", "seed", "early_stop_patience")}
        lines.append(f"## Trial {trial_idx}  (objective = {r.get('objective', float('nan')):.4e})")
        lines.append("")
        lines.append(f"- Test 1 one-step MAE: `{r.get('mae_t1_one_step', float('nan')):.4e}`")
        lines.append(f"- Test 2 one-step MAE: `{r.get('mae_t2_one_step', float('nan')):.4e}`")
        lines.append(f"- Train time: `{r.get('train_time_s', float('nan')):.1f}s`")
        lines.append("")
        lines.append("**Strategy agent rationale:**")
        lines.append("")
        lines.append("> " + r["llm_rationale"].replace("\n", "\n> "))
        lines.append("")
        lines.append("**Tuning agent config:**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(hp, indent=2, default=str))
        lines.append("```")
        lines.append("")

    out_path = Path(args.out) if args.out else (sd / "rationales.md")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(chosen)} rationales -> {out_path}")


if __name__ == "__main__":
    main()
