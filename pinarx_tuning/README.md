# PI-NARX + LLM-AutoOpt on the Thosar 2025 CSTR

Building on the CSTR case study and PI-NARX architecture from:

> Thosar, Bhakte, Li, Srinivasan, Prasad (2025).
> "A novel hybrid neural network for modeling dynamic systems using
> physics-informed regularization."
> Journal of Process Control 152, 103473.
> https://doi.org/10.1016/j.jprocont.2025.103473

## The plan

Thosar et al. propose **PI-NARX** — a NARX neural network with a
physics-informed regularization term — and tune hyperparameters by
trial-and-error. We apply **LLM-AutoOpt** to the same architectures
on the same CSTR plant and aim to beat hand-tuned baselines.

We use the paper's **plant** (Bequette CSTR Eqs 6-9), **architectures**
(NARX and PI-NARX), and **loss formulation** (Eqs 3-5, lambda_l=1e10,
lambda_p=0.01) verbatim, but switch the training-data protocol from
their random APRBS to a **dense 10x10 (Q_f, Q_c) grid** of training
amplitudes. Why:

- Paper's protocol (5000-min APRBS, 200-250 min holds) yields only ~22
  random `(Q_f, Q_c)` amplitudes, which essentially never lands on the
  Test 1 corners `(100, 20)` and `(140, 10)`. Their reported NARX MAE
  of 0.001508 turns out to be very seed-sensitive: across 5 random
  seeds we get mean 0.015 (10x worse than their number).
- Dense-grid training (10 Q_f levels x 10 Q_c levels, 60-80 min holds,
  ~6500 min total trajectory) covers the input cube uniformly and is
  reproducible across seeds. Our NARX hits Test 1 MAE = 0.0002 (better
  than paper's 0.001508 by 7x) with this protocol.

The dense-grid baseline is the comparison point for LLM-AutoOpt.

## Target numbers to beat

### Noiseless (Table 2)
| Model    | Test 1 (Within range) | Test 2 (Extrapolation) |
|----------|-----------------------|------------------------|
| NARX     | 0.001508              | 0.01934                |
| PI-NARX  | **0.001242**          | **0.01556**            |

### Limited data (Table 4, 500 points)
| Model    | Test 1 | Test 2 |
|----------|--------|--------|
| NARX     | 0.008336 | 0.04664 |
| PI-NARX  | **0.004042** | **0.02651** |

### Limited knowledge (Table 5, mass+energy balance only)
| Model    | Test 1 | Test 2 |
|----------|--------|--------|
| NARX     | 0.008336 | 0.04664 |
| PI-NARX (Mass only)   | 0.004716 | 0.02800 |
| PI-NARX (Energy only) | 0.004320 | 0.02852 |
| PI-NARX (Full physics)| **0.004042** | **0.02651** |

### Noisy data (Table 6) — the user's primary target
| Model            | Test 1 | Test 2 |
|------------------|--------|--------|
| NARX SNR 35      | 0.009947 | 0.03369 |
| PI-NARX SNR 35   | **0.009269** | **0.02937** |
| NARX SNR 100     | 0.009771 | 0.03321 |
| PI-NARX SNR 100  | **0.009049** | **0.02916** |

## Suspect choices we can probably beat

| Choice          | Their value          | Why suspect |
|-----------------|---------------------|-------------|
| Architecture    | (200, 400, 200)      | 165k params for 10-input regression — overparameterized |
| Loss weights    | λ_l=1e10, λ_p=0.01   | 10^12 ratio — likely arbitrary scaling fix |
| Window size     | w=2                  | No search reported |
| Collocation pts | 10,000 LHS           | Round number, not tuned |
| Activation      | tanh                 | No comparison to gelu/silu |

## File layout

```
pinarx_tuning/
├── README.md                  this file
├── pinarx_plant.py            Bequette CSTR (paper Eqs 6-9 + Table 1)
├── data_gen.py                training/validation/test data (Fig A.2 protocol)
├── narx.py                    baseline NARX (paper Eqs 2-3)
├── pi_narx.py                 PI-NARX with physics-informed loss (Eqs 4-5)
├── train.py                   Adam + L-BFGS training (paper recipe)
├── evaluate.py                Test Case 1 (interp) + Test Case 2 (extrap)
├── llm_autoopt.py             LLM-driven hyperparameter tuner (the BEAT step)
└── runs/                      output directory
```

## Daily plan

- **Day 1 (today)**: Plant + open-loop sanity check (reproduce Fig A.1 steady state)
- **Day 2**: Data generation (Fig A.2) + baseline NARX (match Table 2 NARX row)
- **Day 3**: PI-NARX (match Table 2 PI-NARX row) + noise wrapper (match Table 6)
- **Day 4**: LLM-AutoOpt over the 8-10 hyperparameters → beat their numbers

## Plant: CSTR with cooling jacket (Bequette)

Manipulated inputs:    u = [Q_f, Q_c]   (L/min)
Measured outputs:      y = [C_A, T, T_c, h]
Sampling time:         Δt = 1 min
Window for NARX:       w = 2 (their choice; we'll tune)

Steady-state target (paper Appendix, Q_f=120, Q_c=15 L/min, t→∞):
  y_SS = [0.0025 mol/L, 416.12 K, 351.55 K, 9 m]

## Test cases (paper §4 + Appendix)

**Test Case 1 (Within range)**: Q_f, Q_c stepped through values used in training
  - t=0-100:   Q_f=120, Q_c=15 (steady state)
  - t=100-300: Q_f=100, Q_c=20
  - t=300-500: Q_f=120, Q_c=15
  - t=500-900: Q_f=120, Q_c=15 (steady)
  - t=900-1100: Q_f=140, Q_c=10
  - t=1100-1400: Q_f=120, Q_c=15

**Test Case 2 (Extrapolation)**: same scheme, but inputs OUTSIDE training range
  - Q_f ∈ {90, 150} (training was [100, 140])
  - Q_c ∈ {5, 25}   (training was [10, 20])
