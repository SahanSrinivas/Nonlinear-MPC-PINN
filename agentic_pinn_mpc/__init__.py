"""Agentic PINN-MPC: extends Kardamaki et al. 2026 (JPC) PINN-MPC with
LLM-AutoOpt hparam tuning, agentic training monitor, and RL refinement.

External code (from https://github.com/ntua-unit-of-control-and-informatics/pinn-mpc)
is in `external/` and licensed CC-BY-NC-SA.

Our extension is in `agentic_pinn_mpc/`.

Reference:
  Kardamaki, A., Protoulis, T., Alexandridis, A., Sarimveis, H. (2026).
  "An explicit model predictive control framework based on physics-informed
   neural networks." J. Process Control 158, 103634.
   https://doi.org/10.1016/j.jprocont.2026.103634
"""
