"""PC-Gym case studies for Paper 3 PINN-MPC + LLM-AutoOpt + DPC extension.

Two new plants ported VERBATIM from Bloor et al. 2025
(Computers and Chemical Engineering 204, 109363):

  - plant_crystallization.py : Section 4.1.3 (K2SO4 cooling crystallization,
    de Moraes et al. 2023 model: 5 ODEs, 1 input, 2 outputs)
  - plant_fourtank.py        : Section 4.1.4 (Johansson 2000 four-tank,
    4 ODEs, 2 inputs, 2 outputs)

All equations + parameters are from the paper, no assumptions made.
See PARAMETERS.md for the full table of values.
"""
