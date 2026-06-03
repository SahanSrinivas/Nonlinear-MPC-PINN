# PC-Gym Case Studies — verbatim parameter & equation tables

All values reproduced exactly from:
> Bloor M., Torraca J., Sandoval I.O., Ahmed A., White M., Mercangöz M., Tsay C.,
> Del Rio Chanona E.A., Mowbray M. (2025). **"PC-Gym: Benchmark environments for
> process control problems."** *Computers and Chemical Engineering* 204, 109363.
> DOI: https://doi.org/10.1016/j.compchemeng.2025.109363

**No assumptions made.** Every value below is from a Table, an Equation, or Section text in the paper.

---

## Case Study 3 — Crystallization reactor (PC-Gym Section 4.1.3)

### Plant: K2SO4 cooling crystallization (de Moraes et al. 2023 model)
- **States**: `[μ_0, μ_1, μ_2, μ_3, c]`
  - μ_0, μ_1, μ_2, μ_3 = first four moments of crystal size distribution
  - c = solute concentration
- **Control input** (1): `T_c` (temperature, °C)
- **Controlled outputs** (2): `CV` (coefficient of variation), `L_n` (number-average crystal size)

### Equations (paper Eqs. 20-30, verbatim)
```
dμ_0/dt = B_0                                                            (20)
dμ_1/dt = G_∞ · (a·μ_0 + b·μ_1·10⁻⁴) · 10⁴                              (21)
dμ_2/dt = 2·G_∞ · (a·μ_1·10⁻⁴ + b·μ_2·10⁻⁸) · 10⁸                       (22)
dμ_3/dt = 3·G_∞ · (a·μ_2·10⁻⁸ + b·μ_3·10⁻¹²) · 10¹²                    (23)
dc/dt   = -0.5·ρ·α·G_∞·(a·μ_2·10⁻⁸ + b·μ_3·10⁻¹²)                       (24)

C_eq = -686.2686 + 3.579165·(T_c + 273.15) - 0.00292874·(T_c + 273.15)² (25)
S    = c · 10³ − C_eq                                                    (26)
B_0  = k_a · exp(k_b/(T_c+273.15)) · (S²)^(k_c/2) · (μ_3²)^(k_d/2)      (27)
G_∞  = k_g · exp(k_1/(T_c+273.15)) · (S²)^(k_2/2)                       (28)

CV   = √(μ_2·μ_0/μ_1² − 1)                                              (29)
L_n  = μ_1 / μ_2          [VERBATIM — paper Eq. (30)]                   (30)
```

### Parameter values (paper Table 3, verbatim)

| Parameter | Description | Value |
|---|---|---|
| `k_a` | Nucleation rate constant | **0.92** |
| `k_b` | Nucleation temperature dependency | **−6800** |
| `k_c` | Nucleation supersaturation exponent | **0.92** |
| `k_d` | Nucleation crystal content exponent | **1.3** |
| `k_g` | Growth rate constant | **48** |
| `k_1` | Growth rate temperature dependency | **−4900** |
| `k_2` | Growth rate supersaturation exponent | **1.9** |
| `a` | Size-dependent growth parameter | **0.51** |
| `b` | Size-dependent growth parameter | **7.3** |
| `α` | Volumetric shape factor | **7.5** |
| `ρ` | Crystal density (g/cm³) | **2.7** |

### Episode + NMPC (paper p.10, Section 4.3.3)

| Item | Value |
|---|---|
| Episode duration | **30 hours** |
| Timesteps per episode | **30** |
| Controller sampling time | 1 hour (= 30 hr / 30 steps) |
| NMPC prediction horizon | **10** |
| NMPC Q matrix | **Identity** |
| NMPC R matrix | **Zero** |

### Bloor et al. reported results (paper Tables 5 & 6)

| Algorithm | Optimality gap | MAD |
|---|---|---|
| DDPG | 0.0212 | 0.0033 |
| SAC | 0.0148 | 0.0009 ← lowest MAD |
| **PPO** | **0.0103** ← lowest gap | 0.0013 |

---

## Case Study 4 — Four-tank multivariable system (PC-Gym Section 4.1.4)

### Plant: quadruple-tank water system (Johansson 2000)
- **States** (4): `[h_1, h_2, h_3, h_4]` water levels in 4 interconnected tanks (m)
- **Control inputs** (2): `[v_1, v_2]` pump voltages (V)
- **Controlled outputs** (2): `h_1, h_2` (lower tank levels)
- **Coupling**: `γ_1`, `γ_2` split ratios — non-minimum phase capable

### Equations (paper Eqs. 31-34, verbatim)
```
dh_1/dt = -(a_1/A_1)·√(2·g_a·h_1) + (a_3/A_1)·√(2·g_a·h_3) + (γ_1·k_1/A_1)·v_1   (31)
dh_2/dt = -(a_2/A_2)·√(2·g_a·h_2) + (a_4/A_2)·√(2·g_a·h_4) + (γ_2·k_2/A_2)·v_2   (32)
dh_3/dt = -(a_3/A_3)·√(2·g_a·h_3) + ((1−γ_2)·k_2/A_3)·v_2                        (33)
dh_4/dt = -(a_4/A_4)·√(2·g_a·h_4) + ((1−γ_1)·k_1/A_4)·v_1                        (34)
```

### Parameter values (paper Table 4, verbatim)

| Parameter | Description | Value |
|---|---|---|
| `g_a` | Acceleration due to gravity (m/s²) | **9.8** |
| `γ_1` | Fraction bypassed by valve to tank 1 | **0.20** |
| `γ_2` | Fraction bypassed by valve to tank 2 | **0.20** |
| `k_1` | 1st pump gain (m³/V·s) | **8.5 × 10⁻⁴** |
| `k_2` | 2nd pump gain (m³/V·s) | **9.5 × 10⁻⁴** |
| `a_1` | Cross-section area of outlet of tank 1 (m²) | **3.5 × 10⁻³** |
| `a_2` | Cross-section area of outlet of tank 2 (m²) | **3.0 × 10⁻³** |
| `a_3` | Cross-section area of outlet of tank 3 (m²) | **2.0 × 10⁻³** |
| `a_4` | Cross-section area of outlet of tank 4 (m²) | **2.5 × 10⁻³** |
| `A_1` | Cross-section area of tank 1 (m²) | **1.0** |
| `A_2` | Cross-section area of tank 2 (m²) | **1.0** |
| `A_3` | Cross-section area of tank 3 (m²) | **1.0** |
| `A_4` | Cross-section area of tank 4 (m²) | **1.0** |

### Episode + NMPC (paper p.10, Section 4.3.4)

| Item | Value |
|---|---|
| Episode duration | **1000 s** |
| Timesteps per episode | **60** |
| Controller sampling time | ≈16.667 s (= 1000/60) |
| NMPC prediction horizon | **17** |
| NMPC Q matrix | **Identity** |
| NMPC R matrix | **Zero** |

### Bloor et al. reported results (paper Tables 5 & 6)

| Algorithm | Optimality gap | MAD |
|---|---|---|
| **DDPG** | **0.0427** ← lowest gap | 0.0980 |
| SAC | 0.0537 | 0.0788 ← lowest MAD |
| PPO | 0.0690 | 0.0994 |

---

## Case Study 1 (already in `agentic_pinn_mpc/`) — Kardamaki single-tank

Already implemented at full precision; see `agentic_pinn_mpc/pinn_siso.py`
and `agentic_pinn_mpc/tuners.py` (`KARDAMAKI_BEST` dict).

---

## What's developed in this folder so far

| File | Status | Source |
|---|---|---|
| `plants/crystallization.py` | ✅ Built + smoke verified | **PC-Gym REPO** values + paper Eqs |
| `plants/fourtank.py` | ✅ Built + smoke verified | **PC-Gym REPO** values + paper Eqs |
| `plants/__init__.py` | ✅ | Re-exports |
| `nmpc_crystallization.py` | ✅ Built + 5-step closed-loop verified | do-mpc N=10, Q=I, R=0 (paper Sec 4.3.3) |
| `nmpc_fourtank.py` | ✅ Built + 20-step closed-loop verified | do-mpc N=17, Q=I, R=0 (paper Sec 4.3.4) |
| `pinn_crystallization.py` | ✅ Architecture + forward pass verified | Kardamaki-style, 9-in/6-out, 9.3k params |
| `pinn_fourtank.py` | ✅ Architecture + forward pass verified | Kardamaki Sec 4.2 MIMO, 9-in/6-out, 9.3k params |
| `pinn_training.py` | ✅ ODE residuals + autograd verified | shared loss machinery |
| `PARAMETERS.md` | ✅ This file | Reference table + repo deltas |
| (next) composite loss + training loops | pending | Kardamaki Phase 1/Phase 2 schedule |
| (next) data generators | pending | sample episodes + query NMPC |
| (next) evaluator | pending | Bloor optimality gap + MAD |
| (next) `tuners.py` + bench | pending | reuses `agentic_pinn_mpc/tuners.py` machinery |
| (next) DPC refinement | pending | adapts `agentic_pinn_mpc/rl_refine.py` |

---

## IMPORTANT: Repo vs Paper deltas

Where the PC-Gym repo (`github.com/MaximilianB2/pc-gym/src/pcgym/model_classes.py`) differs from the published paper, we use the **repo** values because the repo IS the code that produced Bloor et al.'s reported optimality gaps:

| Param | Paper Table | Repo (used here) | Notes |
|---|---|---|---|
| Cryst `k_a` | 0.92 | **0.923714966** | rounded in paper |
| Cryst `k_b` | -6800 | **-6754.878558** | rounded |
| Cryst `k_c` | 0.92 | **0.92229965554** | rounded |
| Cryst `k_d` | 1.3 | **1.341205945** | rounded |
| Cryst `k_g` | 48 | **48.07514464** | rounded |
| Cryst `k_1` | -4900 | **-4921.261419** | rounded |
| Cryst `k_2` | 1.9 | **1.871281405** | rounded |
| Cryst `a` | 0.51 | **0.50523693** | rounded |
| Cryst `b` | 7.3 | **7.271241375** | rounded |
| Cryst `α` | 7.5 | **7.510905767** | rounded |
| Cryst `ρ` | 2.7 | **2.658** | rounded |
| Cryst `L_n` formula | μ_1/μ_2 (Eq 30) | **μ_1/μ_0** | **paper typo** — repo uses standard definition |
| Four-tank `g_a` | 9.8 | **9.81** | rounded |

Standard repo values everywhere else (γ, k_1, k_2, a_i, A_i for four-tank match paper Table 4 exactly).
