"""Plant models for the PC-Gym case studies.

Verbatim ports from Bloor et al. 2025 (Comp & Chem Eng 204, 109363):
  - crystallization.py : Case Study 3 (K2SO4 cooling, 5 ODEs)
  - fourtank.py        : Case Study 4 (4-tank multivariable, 4 ODEs)

NO ASSUMPTIONS - all equations and parameter values exactly as printed
in the paper. See ../PARAMETERS.md for the full table.
"""
from .crystallization import (
    CrystParams, CrystScenario,
    rhs as crystallization_rhs,
    step as crystallization_step,
    C_eq, CV_from_moments, Ln_from_moments,
)
from .fourtank import (
    FourTankParams, FourTankScenario,
    rhs as fourtank_rhs,
    step as fourtank_step,
)

__all__ = [
    "CrystParams", "CrystScenario",
    "crystallization_rhs", "crystallization_step",
    "C_eq", "CV_from_moments", "Ln_from_moments",
    "FourTankParams", "FourTankScenario",
    "fourtank_rhs", "fourtank_step",
]
