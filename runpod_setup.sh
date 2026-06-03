#!/bin/bash
# RunPod deployment script for Paper 3 PINN-MPC + Agentic Stack
# Single-tank water-level case study (Kardamaki 2026 baseline).
#
# Flow (per user direction):
#   1. Sanity test (small scale, ~30s)
#   2. LLM-AutoOpt full bench (~3-4 hr) — Phase A hparam search
#   3. Run the FULL PIPELINE on LLM's best config:
#        - Train PINN-MPC with LLM-AutoOpt-best hparams (Phase A)
#        - Apply DPC refinement on top (Phase C)
#        - Closed-loop evaluation
#   4. Compare against Kardamaki 2026 published baseline (mean offset ~0.034)
#   5. (Later, optional) BO + Optuna for head-to-head comparison
#       — Random is SKIPPED per user direction
#
# Benchmark reference: Kardamaki et al. 2026 Table 4 (SISO single-tank):
#   mean tracking offset    = 0.0161 m
#   mean disturbance offset = 0.0129 m
#   max  tracking offset    = 0.0322 m
#   max  disturbance offset = 0.0453 m
#   -> combined_score (our metric) ~= 0.0339

set -e

echo "=== Paper 3 / Agentic PINN-MPC RunPod bench ==="
echo "GPU:"
nvidia-smi -L || echo "  WARNING: no GPU (will fall back to CPU)"
echo ""

# ---------- 0. Setup ----------
echo "Installing Python deps..."
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -d "external" ]; then
    echo "Cloning Kardamaki PINN-MPC external repo for training data..."
    git clone https://github.com/ntua-unit-of-control-and-informatics/pinn-mpc external
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set."
    echo "  export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi

if [ ! -f "external/siso_training_samples.pt" ]; then
    echo "ERROR: external/siso_training_samples.pt missing."
    exit 1
fi

mkdir -p results

# ---------- 1. Sanity test ----------
echo ""
echo "=== STEP 1: Sanity test (1 LLM trial, K=200, ~1 min) ==="
python -u -m agentic_pinn_mpc.bench \
    --tuners llm \
    --n-trials 1 \
    --K1 200 --K2 200 --bs 50 \
    --n-eval-tracking 100 --n-eval-disturbance 100 \
    --output results/sanity_test \
    2>&1 | tee results/sanity_test.log
echo "Sanity test complete."
echo ""

# ---------- 2. LLM-AutoOpt full bench (Phase A) ----------
echo "=== STEP 2: LLM-AutoOpt full bench (Phase A hparam search) ==="
echo "Expected: ~3-4 hours on RTX PRO 6000 / ~5-7 hours on T4"
echo "  25 trials, K1=K2=10000, bs=100, eval on 500 tracking + 500 disturbance"
echo "  Per-trial checkpoints in results/llm_phaseA/"
echo ""
echo "Press Ctrl+C in next 10 seconds to cancel..."
sleep 10

python -u -m agentic_pinn_mpc.bench \
    --tuners llm \
    --n-trials 25 \
    --K1 10000 --K2 10000 --bs 100 \
    --n-eval-tracking 500 --n-eval-disturbance 500 \
    --output results/llm_phaseA \
    --seed 0 \
    2>&1 | tee results/llm_phaseA.log

echo ""
echo "=== STEP 2 done. LLM-AutoOpt Phase A result ==="
python -c "
import json
with open('results/llm_phaseA/llm_trials.json') as f: d=json.load(f)
print(f'  best score:      {d[\"best_score\"]:.6f}')
print(f'  best cfg:        {d[\"best_cfg\"]}')
print(f'  total trials:    {len(d[\"trials\"])}')
print()
KARDAMAKI_REF = 0.0339  # combined_score from published Table 4 offsets
print(f'  Kardamaki 2026 published combined_score: ~{KARDAMAKI_REF:.4f}')
print(f'  LLM-AutoOpt best vs published:           '
      f'{d[\"best_score\"]/KARDAMAKI_REF:.2f}x')
"
echo ""

# ---------- 3. Full pipeline on LLM's best (Phase A + Phase C DPC refinement) ----------
echo "=== STEP 3: Full pipeline on LLM's best config ==="
echo "  Trains PINN with LLM-AutoOpt best hparams, then applies DPC refinement"
echo "  Expected: ~10-20 min on GPU"
echo ""
sleep 5

python <<'EOF'
import json
import time
import torch

from agentic_pinn_mpc.bench import load_training_data
from agentic_pinn_mpc.pinn_siso import (PINNHparams, train_pinn_siso, DEVICE)
from agentic_pinn_mpc.evaluate import evaluate_model, EvalScenario
from agentic_pinn_mpc.rl_refine import refine_with_dpc, DPCRefineCfg

# Load LLM's best
with open('results/llm_phaseA/llm_trials.json') as f:
    llm = json.load(f)
best_cfg = llm['best_cfg']
print(f"LLM-AutoOpt best cfg: {best_cfg}")
print()

# Load training data
print("Loading training data...")
x0_all, u0_all, ysp_all, d0_all = load_training_data()
print(f"  {x0_all.shape[0]} episodes")
print()

# 3a) Train PINN with LLM best hparams (Phase A re-train at full precision)
hp = PINNHparams(
    w_ode=best_cfg['w_ode'], w_ic=best_cfg['w_ic'],
    w_ytrk=best_cfg['w_ytrk'], w_utrk=best_cfg['w_utrk'],
    w_du=best_cfg['w_du'], w_u=best_cfg['w_u'], w_x=best_cfg['w_x'],
    lr1=best_cfg['lr1'], lr2=best_cfg['lr2'],
    K1=10000, K2=10000, bs=100,
)
print("Training PINN with LLM-best hparams (K1=K2=10000)...")
t0 = time.time()
model, _ = train_pinn_siso(hp, x0_all, u0_all, ysp_all, d0_all, verbose=False)
print(f"  Train time: {time.time()-t0:.1f}s")

# 3b) Evaluate BEFORE refinement
print("\nClosed-loop eval (BEFORE DPC refinement, 500+500 episodes):")
m_before = evaluate_model(model, n_tracking=500, n_disturbance=500,
                            use_kardamaki_samples=True)
for k in ["tracking_mean_offset_m", "tracking_max_offset_m",
          "disturbance_mean_offset_m", "disturbance_max_offset_m",
          "combined_score"]:
    print(f"  {k}: {m_before[k]:.4f}")

# 3c) Phase C: DPC refinement
print("\nPhase C: DPC refinement (200 epochs, lr=1e-5)...")
cfg = DPCRefineCfg(epochs=200, bs=32, lr=1e-5)
t0 = time.time()
model_refined, hist = refine_with_dpc(model, cfg, verbose=True)
print(f"  Refine time: {time.time()-t0:.1f}s")

# 3d) Evaluate AFTER refinement
print("\nClosed-loop eval (AFTER DPC refinement):")
m_after = evaluate_model(model_refined, n_tracking=500, n_disturbance=500,
                           use_kardamaki_samples=True)
for k in ["tracking_mean_offset_m", "tracking_max_offset_m",
          "disturbance_mean_offset_m", "disturbance_max_offset_m",
          "combined_score"]:
    before = m_before[k]
    after = m_after[k]
    delta = (after - before) / max(1e-9, before) * 100
    print(f"  {k}: {before:.4f} -> {after:.4f}  ({delta:+.1f}%)")

# Save
with open('results/llm_phaseA/full_pipeline_result.json', 'w') as f:
    json.dump({
        "llm_best_cfg":     best_cfg,
        "metrics_before_refine": m_before,
        "metrics_after_refine":  m_after,
    }, f, indent=2)

# ---- 4. Comparison vs Kardamaki published ----
print()
print("=" * 60)
print("=== STEP 4: COMPARISON vs Kardamaki 2026 published ===")
print("=" * 60)
# Kardamaki Table 4
KARD = {
    "tracking_mean_offset_m":   0.0161,
    "tracking_max_offset_m":    0.0322,
    "disturbance_mean_offset_m":0.0129,
    "disturbance_max_offset_m": 0.0453,
}
KARD_combined = (0.5*(KARD["tracking_mean_offset_m"]
                       + KARD["disturbance_mean_offset_m"])
                  + 0.25*(KARD["tracking_max_offset_m"]
                          + KARD["disturbance_max_offset_m"]))
print(f"\n{'metric':<32}{'Kardamaki':>14}{'LLM only':>12}{'LLM+DPC':>12}{'vs Kard':>10}")
print("-" * 80)
for k in ["tracking_mean_offset_m", "tracking_max_offset_m",
          "disturbance_mean_offset_m", "disturbance_max_offset_m"]:
    kv = KARD[k]
    lv = m_before[k]
    dv = m_after[k]
    print(f"  {k:<30}{kv:>14.4f}{lv:>12.4f}{dv:>12.4f}{dv/kv:>9.2f}x")
print(f"  {'combined_score (our metric)':<30}{KARD_combined:>14.4f}"
      f"{m_before['combined_score']:>12.4f}{m_after['combined_score']:>12.4f}"
      f"{m_after['combined_score']/KARD_combined:>9.2f}x")
print()
if m_after['combined_score'] < KARD_combined:
    delta = (KARD_combined - m_after['combined_score']) / KARD_combined * 100
    print(f"  >>> LLM+DPC BEATS Kardamaki by {delta:.1f}%  <<<")
else:
    delta = (m_after['combined_score'] - KARD_combined) / KARD_combined * 100
    print(f"  >>> LLM+DPC is {delta:.1f}% above Kardamaki  <<<")
EOF

# ---------- 5. (Optional) BO + Optuna later ----------
echo ""
echo "=== Pipeline complete for Phase A (LLM) + Phase C (DPC) ==="
echo ""
echo "To run BO + Optuna baselines later for comparison, run:"
echo "  python -u -m agentic_pinn_mpc.bench \\"
echo "      --tuners bo optuna \\"
echo "      --n-trials 25 \\"
echo "      --K1 10000 --K2 10000 --bs 100 \\"
echo "      --n-eval-tracking 500 --n-eval-disturbance 500 \\"
echo "      --output results/baselines_later"
echo ""
echo "Results so far:"
ls -la results/llm_phaseA/ 2>/dev/null
