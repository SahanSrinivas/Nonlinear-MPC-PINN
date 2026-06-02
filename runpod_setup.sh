#!/bin/bash
# RunPod deployment script for Paper 3 Phase A+B PINN-MPC tuner bench.
#
# Usage on RunPod (RTX PRO 6000 / RTX 6000 Ada / A10G / similar):
#   1. SSH into your RunPod instance
#   2. git clone <this-repo>
#   3. cd Nonlinear-LLMs-PINN-MPC
#   4. export ANTHROPIC_API_KEY=sk-ant-...
#   5. bash runpod_setup.sh
#
# The script installs deps, clones the external Kardamaki repo (for data),
# and runs the Phase A bench. Final results land in results/runpod_full/.

set -e

echo "=== Paper 3 / RunPod setup ==="
echo "GPU:"
nvidia-smi -L || echo "  WARNING: no GPU detected (will fall back to CPU)"
echo ""

# 1. Python deps
echo "Installing Python deps..."
pip install --upgrade pip
pip install -r requirements.txt

# 2. Clone Kardamaki external repo for training data (if not already present)
if [ ! -d "external" ]; then
    echo "Cloning Kardamaki PINN-MPC repo for training data..."
    git clone https://github.com/ntua-unit-of-control-and-informatics/pinn-mpc external
fi

# 3. Verify API key + dataset
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi
if [ ! -f "external/siso_training_samples.pt" ]; then
    echo "ERROR: external/siso_training_samples.pt missing."
    exit 1
fi

# 4. Quick CPU/GPU sanity test (2 trials of Random, very small scale)
echo ""
echo "=== Sanity test (2 trials, ~30s) ==="
python -u -m agentic_pinn_mpc.bench \
    --tuners random \
    --n-trials 2 \
    --K1 200 --K2 200 --bs 50 \
    --n-eval-tracking 100 --n-eval-disturbance 100 \
    --output results/sanity_test

# 5. Full bench (all 4 tuners, full Kardamaki scale)
echo ""
echo "=== Phase A FULL BENCH (RunPod, all 4 tuners, full scale) ==="
echo "Expected: ~5 hours on RTX PRO 6000"
echo "Press Ctrl+C in next 10 seconds to cancel..."
sleep 10

python -u -m agentic_pinn_mpc.bench \
    --tuners random bo optuna llm \
    --n-trials 25 \
    --K1 10000 --K2 10000 --bs 100 \
    --n-eval-tracking 500 --n-eval-disturbance 500 \
    --output results/runpod_full \
    --seed 0 \
    2>&1 | tee results/runpod_full.log

echo ""
echo "=== Done. Results: results/runpod_full/ ==="
ls -la results/runpod_full/
