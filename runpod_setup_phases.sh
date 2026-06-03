#!/bin/bash
# Three-phase improvement plan over the original runpod_setup.sh.
# Run after STEP 2 of runpod_setup.sh has finished (results/llm_phaseA/
# exists and llm_trials.json has a best_cfg).
#
# Phase 1 (~25 min): two-phase DPC + importance weighting on LLM-best
# Phase 2 (~1.5 hr): multi-objective BO over the same 9-D HSPACE
# Phase 3 (~25 min): LEAN 3-agent tuner (~15 trials with diagnostic feedback)
#
# Each phase is independent - skip any by commenting out its block.

set -e

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "WARNING: ANTHROPIC_API_KEY unset (only Phase 3 LEAN tuner uses LLM in some modes)"
fi

mkdir -p results

# Make sure pyyaml is available for the LEAN tuner
pip install --quiet pyyaml pymoo

# ============================================================
# Phase 1: two-phase DPC + importance weighting
# ============================================================
echo ""
echo "=========================================================="
echo "=== PHASE 1: two-phase DPC + importance weighting ========"
echo "=========================================================="
echo "Trains PINN with LLM-best cfg + importance_d0_alpha=1.0,"
echo "then DPC-refines tracking (50 ep) then disturbance (200 ep)."
echo ""
python -u phase1_two_phase_dpc.py 2>&1 | tee results/phase1.log
echo ""
echo "Phase 1 complete. See results/phase1_two_phase_dpc.json"

# ============================================================
# Phase 2: multi-objective BO
# ============================================================
echo ""
echo "=========================================================="
echo "=== PHASE 2: multi-objective BO over the 9-D HSPACE ======"
echo "=========================================================="
echo "Same 25-trial budget as the LLM tuner, but optimizes the"
echo "4 raw Kardamaki metrics directly (Pareto front)."
echo ""
python -u phase2_mobo.py \
    --n-trials 25 \
    --K1 10000 --K2 10000 --bs 100 \
    --n-eval-tracking 500 --n-eval-disturbance 500 \
    --seed 0 \
    --output results/mobo_phaseA \
    2>&1 | tee results/phase2.log
echo ""
echo "Phase 2 complete. See results/mobo_phaseA/mobo_trials.json"

# ============================================================
# Phase 3: LEAN 3-agent tuner
# ============================================================
echo ""
echo "=========================================================="
echo "=== PHASE 3: LEAN 3-agent tuner ==========================="
echo "=========================================================="
echo "Diagnostic + Strategy + Tuning agents with playbook from"
echo "agentic_pinn_mpc/ref_pinn_fixes.yaml. 15 trials (fewer than"
echo "LLM because the playbook converges faster)."
echo ""
python -u phase3_lean.py \
    --n-trials 15 \
    --K1 10000 --K2 10000 --bs 100 \
    --n-eval-tracking 500 --n-eval-disturbance 500 \
    --seed 0 \
    --output results/lean_phaseA \
    2>&1 | tee results/phase3.log
echo ""
echo "Phase 3 complete. See results/lean_phaseA/lean_trials.json"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================================="
echo "=== ALL PHASES DONE - paper ablation table data ready ===="
echo "=========================================================="
python <<'EOF'
import json
import os

results = {}
for name, path in [
    ("LLM-AutoOpt (original)", "results/llm_phaseA/llm_trials.json"),
    ("LLM+DPC (Phase 1)",      "results/phase1_two_phase_dpc.json"),
    ("MOBO (Phase 2)",         "results/mobo_phaseA/mobo_trials.json"),
    ("LEAN 3-agent (Phase 3)", "results/lean_phaseA/lean_trials.json"),
]:
    if not os.path.exists(path):
        results[name] = None
        continue
    with open(path) as f:
        d = json.load(f)
    if "best_score" in d:
        results[name] = d["best_score"]
    elif "metrics_after_disturbance_dpc" in d:
        results[name] = d["metrics_after_disturbance_dpc"]["combined_score"]
    else:
        results[name] = None

print(f"\n{'Method':<28}{'combined_score':>16}{'vs Kardamaki':>16}")
print("-" * 60)
KARD = 0.0339
print(f"{'Kardamaki 2026 (paper)':<28}{KARD:>16.4f}{'1.00x':>16}")
for name, score in results.items():
    if score is None:
        print(f"{name:<28}{'N/A':>16}{'-':>16}")
    else:
        ratio = score / KARD
        verdict = f"{ratio:.2f}x" + (" WIN" if score < KARD else "")
        print(f"{name:<28}{score:>16.4f}{verdict:>16}")
print()
EOF
