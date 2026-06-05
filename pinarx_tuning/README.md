# Res-Phys NARX on the Thosar 2025 CSTR + LEAN tuner

A residual-physics NARX neural network for the Bequette CSTR case study
from Thosar et al. 2025, tuned by a LEAN-style 3-agent LLM optimizer.

## What this is

- **Plant**: Bequette CSTR with cooling jacket from Thosar et al. 2025
  *Journal of Process Control* 152, 103473 (Eqs 6–9, with three Table 1
  typo corrections detailed in `pinarx_plant.py`).
- **Test cases**: paper-defined Test Case 1 (interpolation) and Test Case 2
  (extrapolation) — see `data_gen.py`.
- **Our model**: **Residual-Physics NARX** — a NARX MLP whose prediction
  is `y_hat = physics_step(y(t-1), u(t-1)) + NN(y_window, u(t-1))`. The
  physics step is a 1-min scipy LSODA integration of the same ODE used
  for the plant; the NN learns only the model-plant mismatch.
- **Tuner**: LEAN-style 3-agent loop (Diagnostic / Strategy / Tuning) +
  Optuna TPE backend, following the Centaur (arXiv:2603.24647) and SLLMBO
  (arXiv:2410.20302) findings that hybrid LLM+TPE beats pure-LLM on
  10-D continuous HPO at small-to-medium budgets.

## What this is NOT

We **do not** try to reproduce Thosar's NARX / PI-NARX numbers from
scratch. Their published Table 2 / Table 6 figures are kept verbatim as
**reference baselines** to compare our Res-Phys NARX against. We had
extensive evidence (see git history) that their reported NARX MAE of
0.001508 is highly seed-sensitive under their stated APRBS protocol —
not worth chasing.

## File layout

```
pinarx_tuning/
├── pinarx_plant.py     Bequette CSTR (paper Eqs 6-9 + 3 Table 1 typo fixes)
├── data_gen.py         Dense-grid training data + Test 1 / Test 2 + noise
├── nn_utils.py         Shared MLP / MinMaxNorm / make_windows
├── resphys_narx.py     THE model: Res-Phys NARX with scipy-LSODA physics
├── train.py            run_trial() - one config -> trained model + metrics
├── evaluate.py         Render paper-comparison table (terminal + LaTeX)
├── README.md           this file
├── runs/               JSON results + tuner logs
└── _archive/           paper-faithful NARX/PI-NARX reproductions
                          (not used; kept for git provenance)
```

## Quick start

```bash
# One training run with default hyperparameters (paper-style architecture)
python train.py --device cuda --seed 0 --out runs/baseline.json

# Render the comparison table
python evaluate.py runs/baseline.json
python evaluate.py runs/baseline.json --latex   # for the paper
```

Expected baseline (10x10 dense grid, default hyperparameters, ~30s GPU):

| Model              | Test 1 (Within range) | Test 2 (Extrapolation) |
|--------------------|-----------------------|------------------------|
| NARX (paper)       | 0.001508              | 0.01934                |
| PI-NARX (paper)    | 0.001242              | 0.01556                |
| **Res-Phys NARX**  | **~0.0002–0.0014**    | **~0.001–0.005**       |

Improvement on Test 2 (extrapolation) is the main story: paper's PI-NARX
struggles outside the training input cube because physics enters only as
a soft regularizer with `lambda_p=0.01`; we get the same physics knowledge
as a hard inductive bias and the NN never has to extrapolate.

## LEAN tuner (in progress)

State-of-the-art research summary (May 2026) lives in `runs/sota_survey.md`.
Headline:

- **Pure LLM optimization** (OPRO, AgentHPO, EvoPrompt, pure-LLM HyperOpt)
  loses to classical TPE/CMA-ES on continuous 10-D HPO once you have
  >50 trials. See [Centaur — arXiv:2603.24647](https://arxiv.org/abs/2603.24647).
- **Hybrid LLM warm-start + TPE/CMA-ES** wins. See
  [LLAMBO — arXiv:2402.03921](https://arxiv.org/abs/2402.03921),
  [SLLMBO — arXiv:2410.20302](https://arxiv.org/abs/2410.20302).
- **Multi-agent PINN tuners** (most relevant precedent):
  [PINNsAgent — arXiv:2501.12053](https://arxiv.org/abs/2501.12053),
  [Lang-PINN — arXiv:2510.05158](https://arxiv.org/abs/2510.05158).

Our design:

```
                  ┌────────────────────┐
   Trial result ─▶│ Diagnostic agent   │  reads MAE per-channel +
                  │ (gpt-4o-mini)      │  classifies failure mode
                  └─────────┬──────────┘
                            ▼
                  ┌────────────────────┐
                  │ Strategy agent     │  emits paragraph-of-rationale +
                  │ (claude-opus-4-8)  │  NL directive ("raise rk4_substeps,
                  │                    │   drop residual_l2")
                  └─────────┬──────────┘
                            ▼
                  ┌────────────────────┐
                  │ Tuning agent       │  converts directive to dict,
                  │ (gpt-4o-mini)      │  pushes to study.enqueue_trial();
                  │                    │  Optuna TPE fills other 70% of trials
                  └─────────┬──────────┘
                            ▼
                       new config →
                       train.run_trial() (~30s GPU)
                       loop
```

LLM consulted on ~30% of trials (Centaur ratio); TPE fills the rest.
At 30s/trial × 150 trials ≈ 75 min wall clock, ~$3–5 in API calls.

## Tunable hyperparameters

The LEAN tuner sweeps the following knobs of `ResPhysNARXHparams`:

| Knob              | Type        | Reasonable range / values         |
|-------------------|-------------|-----------------------------------|
| `hidden`          | tuple[int]  | (50,)..(200,400,200)..(400,800,400) |
| `activation`      | categorical | tanh / relu / gelu / silu         |
| `lr_adam`         | float       | 1e-5 .. 1e-2 (log)                |
| `lr_lbfgs`        | float       | 1e-3 .. 1.0 (log)                 |
| `n_epochs_adam`   | int         | 100 .. 2000                       |
| `lbfgs_iters`     | int         | 100 .. 2000                       |
| `batch_size`      | int         | 16, 32, 64, 128                   |
| `residual_l2`     | float       | 0.0 .. 0.1 (with 0 allowed)       |
| `rk4_sub_steps`   | int         | 20 .. 200 (only used if torch RK4 path is swapped in) |
| `window`          | int         | 1, 2, 3                           |

## References

- Thosar et al. 2025, *J. Process Control* 152, 103473
- Bequette CSTR — Johannesmeyer 2002, *AIChE J.* 48, 2022
- LLM-as-optimizer / LEAN tuner SOTA — see `runs/sota_survey.md`
