"""LEAN-LLM-OPT-style 3-agent tuner for PINN-MPC hyperparameters.

Adapts the structured agentic workflow from
  Liang, K. et al. 2026. "Large-Scale Optimization Model Auto-Formulation:
  Harnessing LLM Flexibility via Structured Workflow." arXiv:2601.09635.

LEAN's three agents (Classification / Workflow Generation / Model Generation)
become here:

  1. Diagnostic agent  - classifies the failure mode from per-metric
                         closed-loop offsets vs Kardamaki Table 4 reference.
                         (Their Classification agent.)

  2. Strategy agent    - retrieves a tuning playbook for the diagnosed
                         mode from `ref_pinn_fixes.yaml`. Optionally
                         calls an LLM as a "senior engineer" to pick the
                         most relevant playbook entry when multiple match.
                         (Their Workflow Generation agent + Ref-Data.)

  3. Tuning agent      - applies the playbook multipliers to the current
                         best config, clips to HSPACE bounds, returns
                         the next config to try. (Their Model Generation
                         agent + tool calls.)

Diagnostic logic:
  - For each metric M in {tracking_mean, tracking_max,
    disturbance_mean, disturbance_max}:
        ratio = obs_M / kard_M
        severity = mild  if ratio < 1.20
                   moderate if 1.20 <= ratio < 1.50
                   severe   if ratio >= 1.50
  - failure_mode = the SINGLE worst metric (by ratio).
  - If all ratios are below tolerance, mode = 'all_in_band'.

The Diagnostic agent is deterministic - no LLM call required.
The Strategy agent is also deterministic when a single playbook entry matches;
it only invokes the LLM when multiple entries could apply (e.g., to break
ties or to compose multi-mode fixes).
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import re
from dataclasses import dataclass

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

from .tuners import HSPACE, KARDAMAKI_BEST, _clip, _load_dotenv_chain


_load_dotenv_chain()


# ---------------------------------------------------------------------------
# Diagnostic agent (deterministic)
# ---------------------------------------------------------------------------
@dataclass
class Diagnosis:
    failure_mode: str       # e.g. 'disturbance_mean_high'
    severity: str           # 'mild' | 'moderate' | 'severe'
    worst_ratio: float      # observed/kardamaki
    per_metric: dict        # all four ratios for transparency


def diagnose(metrics: dict, kardamaki_ref: dict,
             mild_thresh: float = 1.20,
             moderate_thresh: float = 1.50) -> Diagnosis:
    metric_to_mode = {
        "tracking_mean_offset_m":   "tracking_mean_high",
        "tracking_max_offset_m":    "tracking_max_high",
        "disturbance_mean_offset_m":"disturbance_mean_high",
        "disturbance_max_offset_m": "disturbance_max_high",
    }
    ratios = {}
    for k, mode in metric_to_mode.items():
        obs = metrics.get(k, None)
        ref = kardamaki_ref.get(k, None)
        if obs is None or ref is None or ref <= 0:
            continue
        ratios[mode] = obs / ref
    if not ratios:
        return Diagnosis("all_in_band", "mild", 1.0, {})
    worst_mode = max(ratios, key=ratios.get)
    worst = ratios[worst_mode]
    if worst < mild_thresh:
        mode = "all_in_band"
        severity = "mild"
    elif worst < moderate_thresh:
        mode = worst_mode
        severity = "moderate" if worst >= 1.20 else "mild"
    else:
        mode = worst_mode
        severity = "severe"
    # Mild boundary: if mode == worst_mode but ratio < 1.20, still call it mild
    if worst < 1.20 and mode != "all_in_band":
        severity = "mild"
    return Diagnosis(mode, severity, worst, ratios)


# ---------------------------------------------------------------------------
# Strategy agent (deterministic playbook lookup + optional LLM tiebreak)
# ---------------------------------------------------------------------------
@dataclass
class Strategy:
    rationale: str
    param_changes: dict     # {hparam_name: multiplier_or_range}
    train_changes: dict     # e.g. {importance_d0_alpha: 1.0}


def _load_playbook(path: str | None = None) -> dict:
    if yaml is None:
        raise RuntimeError(
            "PyYAML required for LEAN tuner: pip install pyyaml")
    if path is None:
        path = pathlib.Path(__file__).parent / "ref_pinn_fixes.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def select_strategy(diag: Diagnosis, playbook: dict) -> Strategy:
    """Find the playbook entry matching (failure_mode, severity).

    Falls back to a relaxed severity match if the exact tier is absent.
    """
    entries = playbook.get("playbook", [])
    # Exact match first
    for e in entries:
        if e["failure_mode"] == diag.failure_mode \
                and e["severity"] == diag.severity:
            return Strategy(
                rationale=e.get("rationale", ""),
                param_changes=e.get("param_changes", {}),
                train_changes=e.get("train_changes", {}),
            )
    # Relax severity
    for sev in ("severe", "moderate", "mild"):
        for e in entries:
            if e["failure_mode"] == diag.failure_mode and e["severity"] == sev:
                return Strategy(
                    rationale=e.get("rationale", "") + " (severity-relaxed)",
                    param_changes=e.get("param_changes", {}),
                    train_changes=e.get("train_changes", {}),
                )
    # Last-resort: all_in_band/mild
    for e in entries:
        if e["failure_mode"] == "all_in_band":
            return Strategy(
                rationale="no exact match, defaulted to explore",
                param_changes=e.get("param_changes", {}),
                train_changes=e.get("train_changes", {}),
            )
    return Strategy("no playbook match", {}, {})


# ---------------------------------------------------------------------------
# Tuning agent (applies multipliers to current best config)
# ---------------------------------------------------------------------------
def apply_strategy(current_cfg: dict, strat: Strategy,
                   rng: np.random.Generator | None = None) -> dict:
    """Apply playbook multipliers to current_cfg. Clips to HSPACE bounds.

    A multiplier can be:
      - float          -> direct scale
      - [lo, hi] list  -> sample uniformly in that range (for 'explore' mode)
    """
    rng = rng or np.random.default_rng()
    new_cfg = dict(current_cfg)
    for name, m in strat.param_changes.items():
        if name not in new_cfg:
            continue
        if isinstance(m, (list, tuple)) and len(m) == 2:
            mult = float(rng.uniform(m[0], m[1]))
        else:
            mult = float(m)
        new_cfg[name] = new_cfg[name] * mult
    return _clip(new_cfg)


# ---------------------------------------------------------------------------
# The LEAN tuner: same ask/tell API as RandomTuner / LLMTuner / BOTuner
# ---------------------------------------------------------------------------
class LeanTuner:
    """3-agent (Diagnostic / Strategy / Tuning) PINN-MPC hyperparameter tuner.

    On `ask()`:
      - If no history yet, returns Kardamaki warm-start.
      - Otherwise, diagnoses the BEST trial's metrics, picks a playbook
        strategy, applies it to the best config, returns the new config.

    On `tell(cfg, score, metrics)`:
      - Stores the trial. Metrics dict is REQUIRED for the diagnostic to
        function; the bench passes it through.

    Optional: train_changes (e.g. importance_d0_alpha) are also captured
    so the bench can read them and adjust PINNHparams before training.
    """
    name = "lean"

    def __init__(self, seed: int = 0, playbook_path: str | None = None,
                 warm_start: bool = True):
        self.rng = np.random.default_rng(seed)
        self.playbook = _load_playbook(playbook_path)
        self.kardamaki_ref = self.playbook.get("kardamaki_reference", {})
        self.history: list = []
        self.warm_start = warm_start
        self._warmed = False
        self._pending_train_changes = {}   # set by latest ask()

    def _best_so_far(self) -> dict | None:
        if not self.history:
            return None
        finite = [h for h in self.history if np.isfinite(h["score"])]
        if not finite:
            return None
        return min(finite, key=lambda h: h["score"])

    def ask(self) -> dict:
        if self.warm_start and not self._warmed:
            self._warmed = True
            self._pending_train_changes = {}
            return _clip(dict(KARDAMAKI_BEST))
        best = self._best_so_far()
        if best is None:
            # All trials failed - explore randomly around Kardamaki
            cfg = {k: v * float(self.rng.uniform(0.7, 1.3))
                   for k, v in KARDAMAKI_BEST.items()}
            self._pending_train_changes = {}
            return _clip(cfg)
        metrics = best.get("metrics") or {}
        diag = diagnose(metrics, self.kardamaki_ref)
        strat = select_strategy(diag, self.playbook)
        new_cfg = apply_strategy(best["cfg"], strat, rng=self.rng)
        # Stash train changes for the bench to read
        self._pending_train_changes = dict(strat.train_changes)
        # Stash diagnostic metadata on the proposed cfg for logging
        new_cfg["__lean_diagnosis__"] = {
            "failure_mode": diag.failure_mode,
            "severity": diag.severity,
            "worst_ratio": diag.worst_ratio,
            "rationale": strat.rationale,
            "train_changes": dict(strat.train_changes),
        }
        return new_cfg

    def tell(self, cfg: dict, score: float, metrics: dict | None = None):
        clean_cfg = {k: v for k, v in cfg.items()
                     if not k.startswith("__")}
        self.history.append({
            "cfg": clean_cfg,
            "score": score,
            "metrics": metrics or {},
        })

    def pop_train_changes(self) -> dict:
        """Bench reads this after ask() to apply non-hparam adjustments
        (e.g., importance_d0_alpha) to PINNHparams before training."""
        return dict(self._pending_train_changes)


# Register with the global TUNERS dict
def _register():
    from . import tuners as _t
    _t.TUNERS["lean"] = LeanTuner


_register()


if __name__ == "__main__":
    # Smoke test: feed mock metrics, see what the diagnostic + strategy say
    print("=== LEAN diagnostic smoke test ===\n")
    tuner = LeanTuner(seed=0)

    # 1. First ask should return warm-start
    cfg1 = tuner.ask()
    print(f"1. warm-start cfg (first ask): w_du={cfg1['w_du']:.2f} "
          f"(Kardamaki=32.52)")

    # 2. Mock a trial where disturbance is the failure mode (similar to our
    #    pre-fix result: dist_mean=0.025 vs Kard 0.013 = 1.94x)
    mock_metrics = {
        "tracking_mean_offset_m": 0.017,
        "tracking_max_offset_m": 0.035,
        "disturbance_mean_offset_m": 0.025,
        "disturbance_max_offset_m": 0.075,
    }
    tuner.tell(cfg1, score=0.045, metrics=mock_metrics)
    diag = diagnose(mock_metrics, tuner.kardamaki_ref)
    print(f"\n2. diagnose() on disturbance-dominant trial:")
    print(f"   failure_mode = {diag.failure_mode}")
    print(f"   severity     = {diag.severity}")
    print(f"   worst_ratio  = {diag.worst_ratio:.2f}")
    print(f"   per_metric   = "
          + ", ".join(f"{k}={v:.2f}" for k, v in diag.per_metric.items()))

    strat = select_strategy(diag, tuner.playbook)
    print(f"\n3. selected strategy:")
    print(f"   rationale     = {strat.rationale}")
    print(f"   param_changes = {strat.param_changes}")
    print(f"   train_changes = {strat.train_changes}")

    # 3. Second ask should return the playbook-modified config
    cfg2 = tuner.ask()
    diag_meta = cfg2.pop("__lean_diagnosis__", {})
    print(f"\n4. next cfg from playbook (w_du, w_utrk, lr2):")
    print(f"   w_du:   {cfg1['w_du']:.2f} -> {cfg2['w_du']:.2f}  "
          f"(expected ~1.6x for moderate disturbance_mean)")
    print(f"   w_utrk: {cfg1['w_utrk']:.2f} -> {cfg2['w_utrk']:.2f}")
    print(f"   lr2:    {cfg1['lr2']:.2e} -> {cfg2['lr2']:.2e}")
    print(f"\n5. train_changes to apply to PINNHparams:")
    print(f"   {tuner.pop_train_changes()}")
