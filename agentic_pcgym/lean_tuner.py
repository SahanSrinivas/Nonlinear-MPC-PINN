"""LEAN-LLM-OPT-style 3-agent tuner for PC-Gym (four-tank + crystallization).

Mirrors agentic_pinn_mpc/lean_tuner.py (SISO) but adapted to PC-Gym signals.

Three agents per ask():
  1. Diagnostic - reads the BEST trial's (optimality_gap, MAD), computes
                  ratios vs the case's reference (Bloor 2025 Table 5+6),
                  returns (failure_mode, severity).
  2. Strategy   - looks up the playbook entry for (failure_mode, severity)
                  in ref_pinn_fixes_pcgym.yaml.
  3. Tuning     - applies the playbook multipliers to the current-best
                  config, clips to the case's HSPACE bounds, returns the
                  next config.

Diagnostic logic:
  gap_ratio = obs.optimality_gap / ref.opt_gap
  mad_ratio = obs.MAD            / ref.MAD
  worst     = max(gap_ratio, mad_ratio)
  if   both > 1.2:                       both_high
  elif gap_ratio > 1.2:                  opt_gap_high
  elif mad_ratio > 1.2:                  MAD_high
  else:                                  all_in_band
  severity = mild | moderate | severe | critical from worst:
            < 1.2 -> mild,  1.2-1.5 -> moderate,
            1.5-3 -> severe, > 3   -> critical

Special case: if the trial NaN'd (score == 1e6), failure_mode='training_unstable',
severity='critical' (relevant for crystallization in particular).

Same ask/tell API as the LLMTuner / BOTuner so it slots into bench.py's
TUNERS dict.
"""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


_NAN_SENTINEL = 1e6   # bench.py returns this score on training NaN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clip(cfg: dict, hspace: dict) -> dict:
    """Project a candidate cfg into the HSPACE box."""
    return {n: max(lo, min(hi, float(cfg.get(n, 0.5 * (lo + hi)))))
             for n, (lo, hi, _) in hspace.items()}


def _load_playbook(path: str | None = None) -> dict:
    if yaml is None:
        raise RuntimeError(
            "PyYAML required for LEAN tuner: pip install pyyaml")
    if path is None:
        path = pathlib.Path(__file__).parent / "ref_pinn_fixes_pcgym.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Diagnostic agent (deterministic)
# ---------------------------------------------------------------------------
@dataclass
class Diagnosis:
    failure_mode: str       # opt_gap_high | MAD_high | both_high
                            #     | all_in_band | training_unstable
    severity: str           # mild | moderate | severe | critical
    gap_ratio: float        # opt_gap / ref.opt_gap
    mad_ratio: float        # MAD     / ref.MAD


def diagnose(metrics: dict, reference: dict, nan: bool = False) -> Diagnosis:
    """Classify failure mode from (optimality_gap, MAD) vs reference."""
    if nan:
        return Diagnosis("training_unstable", "critical", float("inf"),
                          float("inf"))
    gap = metrics.get("optimality_gap")
    mad = metrics.get("MAD")
    ref_gap = reference.get("opt_gap", None)
    ref_mad = reference.get("MAD", None)
    if gap is None or mad is None or ref_gap in (None, 0) or ref_mad in (None, 0):
        return Diagnosis("all_in_band", "mild", 1.0, 1.0)
    gap_r = float(gap) / float(ref_gap)
    mad_r = float(mad) / float(ref_mad)
    worst = max(gap_r, mad_r)

    # Severity from worst ratio
    if worst < 1.2:
        severity = "mild"
    elif worst < 1.5:
        severity = "moderate"
    elif worst < 3.0:
        severity = "severe"
    else:
        severity = "critical"

    # Mode from which axis is over tolerance
    over_gap = gap_r > 1.2
    over_mad = mad_r > 1.2
    if over_gap and over_mad:
        mode = "both_high"
    elif over_gap:
        mode = "opt_gap_high"
    elif over_mad:
        mode = "MAD_high"
    else:
        mode = "all_in_band"
    return Diagnosis(mode, severity, gap_r, mad_r)


# ---------------------------------------------------------------------------
# Strategy agent (deterministic playbook lookup)
# ---------------------------------------------------------------------------
@dataclass
class Strategy:
    rationale: str
    param_changes: dict


def select_strategy(diag: Diagnosis, case_playbook: dict) -> Strategy:
    """Find playbook entry matching (failure_mode, severity), relax if absent."""
    entries = case_playbook.get("playbook", [])

    # Exact match
    for e in entries:
        if (e["failure_mode"] == diag.failure_mode
                and e["severity"] == diag.severity):
            return Strategy(e.get("rationale", ""),
                              e.get("param_changes", {}))

    # Relax severity, same mode (prefer closer tiers)
    pref_order = {
        "critical":  ["severe", "moderate", "mild"],
        "severe":    ["moderate", "critical", "mild"],
        "moderate":  ["severe", "mild", "critical"],
        "mild":      ["moderate", "severe", "critical"],
    }
    for sev in pref_order.get(diag.severity, ["severe", "moderate", "mild"]):
        for e in entries:
            if e["failure_mode"] == diag.failure_mode and e["severity"] == sev:
                return Strategy(
                    e.get("rationale", "") + f" (severity-relaxed: {sev})",
                    e.get("param_changes", {}))

    # Last-resort: all_in_band/mild
    for e in entries:
        if e["failure_mode"] == "all_in_band":
            return Strategy("no exact match -> explore",
                              e.get("param_changes", {}))

    return Strategy("no playbook match", {})


# ---------------------------------------------------------------------------
# Tuning agent (apply multipliers to current best cfg, clip to bounds)
# ---------------------------------------------------------------------------
def apply_strategy(current_cfg: dict, strat: Strategy, hspace: dict,
                    rng: np.random.Generator | None = None,
                    scalar_jitter: float = 0.20) -> dict:
    """Apply playbook multipliers; clip to HSPACE.

    scalar_jitter: for scalar multipliers (e.g., 2.0), sample uniformly in
    [m * (1 - jitter), m * (1 + jitter)]. Default 0.20 = ±20%. Prevents
    LEAN from proposing the IDENTICAL cfg on consecutive iters when the
    diagnosis doesn't change (which would waste compute on duplicate trials).
    List/tuple multipliers [lo, hi] are unchanged (already-jittered ranges).
    """
    rng = rng or np.random.default_rng()
    new_cfg = dict(current_cfg)
    for name, m in strat.param_changes.items():
        if name not in new_cfg:
            continue
        if isinstance(m, (list, tuple)) and len(m) == 2:
            mult = float(rng.uniform(m[0], m[1]))
        else:
            m_val = float(m)
            lo = m_val * (1.0 - scalar_jitter)
            hi = m_val * (1.0 + scalar_jitter)
            mult = float(rng.uniform(lo, hi))
        new_cfg[name] = new_cfg[name] * mult
    return _clip(new_cfg, hspace)


# ---------------------------------------------------------------------------
# The LEAN tuner: same ask/tell API as RandomTuner / LLMTuner / BOTuner
# ---------------------------------------------------------------------------
class LeanTunerPCGym:
    """3-agent (Diagnostic / Strategy / Tuning) PC-Gym PINN hparam tuner.

    On first ask(), returns the warm-start defaults (Path-1 lowered values).
    On subsequent asks, diagnoses the best trial's (opt_gap, MAD) vs reference,
    looks up a playbook entry, applies it.

    Caller passes:
      hspace   - case's HSPACE dict (FT_HSPACE or CRYST_HSPACE)
      seed     - rng seed
      defaults - case's warm-start cfg (FT_DEFAULT or CRYST_DEFAULT)
      case     - 'fourtank' or 'crystallization' (selects playbook section)
    """
    name = "lean"

    def __init__(self, hspace: dict, seed: int = 0,
                 defaults: dict | None = None,
                 case: str = "fourtank",
                 playbook_path: str | None = None):
        self.hspace = hspace
        self.rng = np.random.default_rng(seed)
        self.defaults = defaults or {}
        self.case = case

        pb = _load_playbook(playbook_path)
        if case not in pb:
            raise ValueError(
                f"LeanTunerPCGym: case '{case}' not in playbook; "
                f"available: {list(pb.keys())}")
        self.case_playbook = pb[case]
        self.reference = self.case_playbook.get("reference", {})

        self.history: list = []
        self._warmed = False
        self._best_score_at_last_ask = float("inf")
        self._stagnation_count = 0       # iters since best last improved
        self._stagnation_threshold = 2   # force explore after this many stale iters

    # -- helpers --
    def _best_so_far(self) -> dict | None:
        if not self.history:
            return None
        finite = [h for h in self.history
                   if np.isfinite(h["score"]) and h["score"] < _NAN_SENTINEL]
        if not finite:
            return None
        return min(finite, key=lambda h: h["score"])

    def _latest_nan(self) -> bool:
        """True if the most recent trial NaN'd (relevant for crystallization)."""
        if not self.history:
            return False
        return self.history[-1]["score"] >= _NAN_SENTINEL

    # -- ask/tell API --
    def ask(self) -> dict:
        # First ask: warm-start
        if not self._warmed and self.defaults:
            self._warmed = True
            return _clip(dict(self.defaults), self.hspace)

        best = self._best_so_far()

        # If everything has NaN'd and there's no best yet, fire the
        # 'training_unstable' playbook entry against the warm-start.
        if best is None:
            diag = Diagnosis("training_unstable", "critical",
                                float("inf"), float("inf"))
            strat = select_strategy(diag, self.case_playbook)
            new_cfg = apply_strategy(self.defaults, strat, self.hspace,
                                         rng=self.rng)
            new_cfg["__lean_diagnosis__"] = {
                "failure_mode": diag.failure_mode,
                "severity": diag.severity,
                "rationale": strat.rationale,
            }
            return new_cfg

        # Track stagnation: did the best improve since last ask?
        if best["score"] < self._best_score_at_last_ask - 1e-6:
            self._stagnation_count = 0
            self._best_score_at_last_ask = best["score"]
        else:
            self._stagnation_count += 1

        # Normal path: diagnose the best, look up strategy, apply
        metrics = {
            "optimality_gap": best.get("optimality_gap"),
            "MAD": best.get("MAD"),
        }
        diag = diagnose(metrics, self.reference,
                          nan=(self._latest_nan() and best is None))

        # Stagnation override: force exploratory move if best hasn't improved
        # in too many iters. Prevents the deterministic playbook from proposing
        # the same cfg over and over.
        forced_explore = self._stagnation_count >= self._stagnation_threshold
        if forced_explore:
            explore_diag = Diagnosis("all_in_band", "mild",
                                            diag.gap_ratio, diag.mad_ratio)
            strat = select_strategy(explore_diag, self.case_playbook)
            self._stagnation_count = 0   # reset after firing explore
            diag_for_log = explore_diag
            rationale_prefix = f"[stagnation-explore] "
        else:
            strat = select_strategy(diag, self.case_playbook)
            diag_for_log = diag
            rationale_prefix = ""

        new_cfg = apply_strategy(best["cfg"], strat, self.hspace,
                                     rng=self.rng)
        new_cfg["__lean_diagnosis__"] = {
            "failure_mode": diag_for_log.failure_mode,
            "severity": diag_for_log.severity,
            "gap_ratio": float(diag.gap_ratio) if np.isfinite(diag.gap_ratio) else None,
            "mad_ratio": float(diag.mad_ratio) if np.isfinite(diag.mad_ratio) else None,
            "rationale": rationale_prefix + strat.rationale,
            "stagnation_count": int(self._stagnation_count),
        }
        return new_cfg

    def tell(self, cfg: dict, score: float, metrics: dict | None = None):
        """Bench passes score; if metrics is None, we extract opt_gap/MAD
        from the trial result dict directly (bench's train_and_score returns
        them alongside score)."""
        clean_cfg = {k: v for k, v in cfg.items()
                       if not k.startswith("__")}
        entry = {"cfg": clean_cfg, "score": float(score)}
        if metrics:
            entry["optimality_gap"] = metrics.get("optimality_gap")
            entry["MAD"] = metrics.get("MAD")
        self.history.append(entry)


# ---------------------------------------------------------------------------
# Self-test: feed mock metrics, print diagnosis + selected strategy
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from .bench import FT_HSPACE, FT_DEFAULT, CRYST_HSPACE, CRYST_DEFAULT

    print("=" * 70)
    print("LEAN tuner smoke test (four-tank)")
    print("=" * 70)
    tuner = LeanTunerPCGym(FT_HSPACE, seed=0, defaults=FT_DEFAULT,
                              case="fourtank")
    cfg0 = tuner.ask()
    print(f"\n1. warm-start cfg: w_ode={cfg0['w_ode']:.2f}  "
          f"w_utrk={cfg0['w_utrk']:.4f}")

    # Mock: trial 1 was the warm-start, scored opt_gap=0.20, MAD=0.15
    tuner.tell(cfg0, score=0.6,
                  metrics={"optimality_gap": 0.20, "MAD": 0.15})
    cfg1 = tuner.ask()
    diag = cfg1.pop("__lean_diagnosis__", {})
    print(f"\n2. after one trial (opt_gap=0.20, MAD=0.15):")
    print(f"   diagnosis: {diag.get('failure_mode')} ({diag.get('severity')})")
    print(f"   gap_ratio: {diag.get('gap_ratio'):.2f}  "
          f"mad_ratio: {diag.get('mad_ratio'):.2f}")
    print(f"   rationale: {diag.get('rationale')}")
    print(f"   new w_ode: {cfg0['w_ode']:.2f} -> {cfg1['w_ode']:.2f}")
    print(f"   new w_du:  {cfg0['w_du']:.2f} -> {cfg1['w_du']:.2f}")

    print("\n" + "=" * 70)
    print("LEAN tuner smoke test (crystallization)")
    print("=" * 70)
    tuner2 = LeanTunerPCGym(CRYST_HSPACE, seed=0, defaults=CRYST_DEFAULT,
                                 case="crystallization")
    cfg0 = tuner2.ask()
    print(f"\n1. warm-start: w_ode={cfg0['w_ode']:.2f}  lr1={cfg0['lr1']:.1e}")

    # Mock NaN'd trial -> training_unstable
    tuner2.tell(cfg0, score=1e6, metrics={})
    cfg_nan = tuner2.ask()
    diag = cfg_nan.pop("__lean_diagnosis__", {})
    print(f"\n2. after NaN trial:")
    print(f"   diagnosis: {diag.get('failure_mode')} ({diag.get('severity')})")
    print(f"   rationale: {diag.get('rationale')}")
    print(f"   new w_ode: {cfg0['w_ode']:.2f} -> {cfg_nan['w_ode']:.2f}")
    print(f"   new lr1:   {cfg0['lr1']:.1e} -> {cfg_nan['lr1']:.1e}")
    print(f"   new lr2:   {cfg0['lr2']:.1e} -> {cfg_nan['lr2']:.1e}")
