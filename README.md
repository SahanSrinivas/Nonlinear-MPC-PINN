# Agentic PINN-MPC — Paper 3

Extends Kardamaki et al. 2026 (J. Process Control, [DOI](https://doi.org/10.1016/j.jprocont.2026.103634), [GitHub](https://github.com/ntua-unit-of-control-and-informatics/pinn-mpc)) with:

- **Phase A**: LLM-AutoOpt for the 9 PINN-MPC hyperparameters (loss weights + learning rates). Head-to-head against their published Optuna result + Random + BO baselines.
- **Phase B**: Agentic training monitor — Claude Sonnet 4.6 watches the loss-component trajectory every N epochs and intervenes (adjust weight, reset Adam, switch phase early, abort).
- **Phase C** *(planned)*: RL refinement (PPO on closed-loop reward after PINN-MPC training).

## Repo layout

```
Nonlinear-LLMs-PINN-MPC/
├── external/                   # Kardamaki's public repo (CC-BY-NC-SA), cloned via git
├── agentic_pinn_mpc/           # OUR extension (this work)
│   ├── pinn_siso.py            # PINN-MPC controller + training (CPU+GPU)
│   ├── evaluate.py             # closed-loop scoring (Kardamaki metric)
│   ├── tuners.py               # Random / BO / Optuna / LLM tuners (common ask/tell API)
│   ├── bench.py                # main entry: runs all 4 tuners, saves results
│   └── monitor.py              # Phase B: LLM training monitor
├── requirements.txt
├── runpod_setup.sh             # one-liner RunPod deployment
└── README.md                   # this file
```

## What's tuned (9 hyperparameters)

Kardamaki et al. 2026 SISO Table 2:

| Weight | Lo bound | Hi bound | Their best | Scale |
|---|---|---|---|---|
| w_ode | 10 | 1000 | 131.20 | log |
| w_ic | 0.1 | 100 | 2.34 | log |
| w_ytrk | 0.5 | 100 | 6.28 | log |
| w_utrk | 0.5 | 100 | 6.73 | log |
| w_du | 1 | 1000 | 32.52 | log |
| w_u | 100 | 10000 | 3243.43 | log |
| w_x | 10 | 1000 | 325.96 | log |
| lr1 | 1e-4 | 1e-2 | 1.01e-3 | log |
| lr2 | 1e-5 | 1e-3 | 2.66e-4 | log |

## Metric

Mean+max steady-state offset across closed-loop test episodes (matches Kardamaki Section 4.1.3 + Table 4). Lower is better. Their published best: 0.016 m mean tracking offset, 0.013 m mean disturbance rejection offset.

## RunPod usage (RTX PRO 6000 recommended)

```bash
# 1. SSH into your RunPod instance with GPU
# 2. Clone this repo
git clone <your-repo-url> && cd Nonlinear-LLMs-PINN-MPC

# 3. Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Run setup + bench (~5 hours)
bash runpod_setup.sh
```

Total compute budget on RTX PRO 6000:
- ~3 min per PINN training trial (Kardamaki full scale: K=10000 epochs/phase, bs=100)
- 25 trials × 4 tuners = 100 trials = ~5 hours
- Plus closed-loop eval: ~10 sec per trial = ~17 min total

**LLM API cost**: ~$1 (Phase A) + ~$7 (Phase B if enabled) with prompt caching.

## Local CPU smoke test (validate before RunPod)

```bash
pip install -r requirements.txt
git clone https://github.com/ntua-unit-of-control-and-informatics/pinn-mpc external
export ANTHROPIC_API_KEY=sk-ant-...

# Reduced-scale smoke test (~1 min per tuner per trial)
python -u -m agentic_pinn_mpc.bench \
    --tuners random llm \
    --n-trials 2 \
    --K1 200 --K2 200 --bs 50 \
    --n-eval-tracking 100 --n-eval-disturbance 100 \
    --output results/cpu_smoke
```

## Citations

- **Original paper**: Kardamaki, Protoulis, Alexandridis, Sarimveis (2026). *An explicit MPC framework based on PINNs.* J. Process Control 158, 103634. https://doi.org/10.1016/j.jprocont.2026.103634
- **Original code**: https://github.com/ntua-unit-of-control-and-informatics/pinn-mpc (CC-BY-NC-SA)
- **PINNs**: Raissi et al. (2019). *Physics-informed neural networks.* J. Comput. Phys. 378, 686-707.
- **LLM-AutoOpt** (this work, persona engineering): see `agentic_pinn_mpc/tuners.py`

## License

This extension is released under CC-BY-NC-SA-4.0 (consistent with the external repo's license).
