"""Four tuners for PINN-MPC hparams, all with a common `ask()` / `tell()` API.

Tuners:
  - random_search: log-uniform sampling
  - bo_search:     scikit-optimize Gaussian-process BO
  - optuna_search: Optuna TPE (Kardamaki et al. 2026 default)
  - llm_search:    Claude Sonnet 4.6 with senior-APC-engineer prompt
                   (+ prompt caching for cost savings)

Hparam space (9-D, log-scale where appropriate):
  Loss weights (7): w_ode, w_ic, w_ytrk, w_utrk, w_du, w_u, w_x
  Learning rates (2): lr1, lr2

Bounds derived from Kardamaki's published-best values, expanded by 10-100x
to give tuners room to search.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# Hparam bounds for tuning. (lo, hi, log_scale)
HSPACE = {
    "w_ode":  (10.0,    1000.0,  True),    # Kardamaki: 131.20
    "w_ic":   (0.1,     100.0,   True),    # Kardamaki: 2.34
    "w_ytrk": (0.5,     100.0,   True),    # Kardamaki: 6.28
    "w_utrk": (0.5,     100.0,   True),    # Kardamaki: 6.73
    "w_du":   (1.0,     1000.0,  True),    # Kardamaki: 32.52
    "w_u":    (100.0,   10000.0, True),    # Kardamaki: 3243.43
    "w_x":    (10.0,    1000.0,  True),    # Kardamaki: 325.96
    "lr1":    (1e-4,    1e-2,    True),    # Kardamaki: 1.01e-3
    "lr2":    (1e-5,    1e-3,    True),    # Kardamaki: 2.66e-4
}
KARDAMAKI_BEST = {
    "w_ode": 131.2036, "w_ic": 2.3417, "w_ytrk": 6.2819,
    "w_utrk": 6.7282, "w_du": 32.5194, "w_u": 3243.4338,
    "w_x": 325.9614, "lr1": 1.008704e-3, "lr2": 2.6554e-4,
}


def _clip(cfg: dict) -> dict:
    out = {}
    for name, (lo, hi, _) in HSPACE.items():
        v = float(cfg.get(name, 0.5 * (lo + hi)))
        out[name] = max(lo, min(hi, v))
    return out


# ---------------------------------------------------------------------------
# Tuner 1: Random search (log-uniform)
# ---------------------------------------------------------------------------
class RandomTuner:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.history: list = []

    def ask(self) -> dict:
        cfg = {}
        for name, (lo, hi, log) in HSPACE.items():
            if log:
                cfg[name] = float(math.exp(
                    self.rng.uniform(math.log(lo), math.log(hi))))
            else:
                cfg[name] = float(self.rng.uniform(lo, hi))
        return _clip(cfg)

    def tell(self, cfg: dict, score: float):
        self.history.append({"cfg": cfg, "score": score})


# ---------------------------------------------------------------------------
# Tuner 2: scikit-optimize Bayesian optimization (GP + LCB)
# ---------------------------------------------------------------------------
class BOTuner:
    name = "bo"

    def __init__(self, seed: int = 0, n_initial: int = 5):
        from skopt import Optimizer
        from skopt.space import Real
        space = []
        for name, (lo, hi, log) in HSPACE.items():
            prior = "log-uniform" if log else "uniform"
            space.append(Real(lo, hi, prior=prior, name=name))
        self.opt = Optimizer(space, base_estimator="GP", random_state=seed,
                              acq_func="LCB", n_initial_points=n_initial)
        self.names = [s.name for s in space]
        self.history: list = []

    def ask(self) -> dict:
        x = self.opt.ask()
        return _clip({n: v for n, v in zip(self.names, x)})

    def tell(self, cfg: dict, score: float):
        x = [cfg[n] for n in self.names]
        self.opt.tell(x, score)
        self.history.append({"cfg": cfg, "score": score})


# ---------------------------------------------------------------------------
# Tuner 3: Optuna TPE (Kardamaki's tuner, for direct head-to-head)
# ---------------------------------------------------------------------------
class OptunaTuner:
    name = "optuna"

    def __init__(self, seed: int = 0):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed))
        self._pending: dict = {}
        self.history: list = []

    def ask(self) -> dict:
        trial = self.study.ask()
        cfg = {}
        for name, (lo, hi, log) in HSPACE.items():
            cfg[name] = trial.suggest_float(name, lo, hi, log=log)
        self._pending[id(cfg)] = trial
        # Hack: stash trial id on cfg via a sentinel key the bench will strip
        cfg["__optuna_trial__"] = trial.number
        return _clip({k: v for k, v in cfg.items()
                       if not k.startswith("__")})

    def tell(self, cfg: dict, score: float):
        # Optuna's ask/tell needs the trial back; we recover by trial.number
        # via a side channel: stash latest trial after ask()
        # Simpler: re-ask each call (since we save state in self.study)
        # Hack — just complete the most recent trial
        trial_number = list(self._pending.values())[-1].number
        self.study.tell(self._pending.popitem()[1], score)
        self.history.append({"cfg": cfg, "score": score,
                              "trial_number": trial_number})


# ---------------------------------------------------------------------------
# Tuner 4: LLM-AutoOpt (Claude Sonnet 4.6, senior-APC-engineer prompt)
# ---------------------------------------------------------------------------
LLM_SYSTEM_PROMPT = """You are a SENIOR APC ENGINEER tuning hyperparameters
for a Physics-Informed Neural Network used as an explicit Model Predictive
Controller (PINN-MPC) on a nonlinear single-tank water level system.

You are configuring the controller from Kardamaki, Protoulis, Alexandridis,
Sarimveis (2026), "An explicit MPC framework based on PINNs", J. Process
Control 158:103634.

PLANT: nonlinear single-tank, dx/dt = (u + d - K*sqrt(x))/A with K=0.7, A=1.
Constraints: x in [0, 4.0] m, u in [0, 1] m^3/s, |du| <= 0.2 per step.

CONTROLLER: a feedforward NN (3 hidden layers 64-16-16) is trained with a
composite loss combining
  - L_ode  (physics: ODE residual)
  - L_ic   (initial-condition consistency)
  - L_ytrk (set-point tracking error)
  - L_utrk (input tracking error against u_ss)
  - L_du   (move suppression via barrier penalty)
  - L_u    (input bound violation)
  - L_x    (state bound violation)

Training is TWO-PHASE:
  Phase 1: only L_ode + L_ic + L_ytrk + L_utrk active (no constraints)
           at learning rate lr1 -> learn dynamics + tracking
  Phase 2: ALL terms active at learning rate lr2 -> refine for constraints

YOU TUNE these 9 hyperparameters (all log-scale):
  w_ode    [10..1000]    typically 100-200 (physics most important)
  w_ic     [0.1..100]    typically 1-10 (modest, IC easy to satisfy)
  w_ytrk   [0.5..100]    typically 5-30 (tracking is core objective)
  w_utrk   [0.5..100]    typically 5-30 (similar to w_ytrk)
  w_du     [1..1000]     typically 10-100 (smoothness)
  w_u      [100..10000]  typically 1000-5000 (high - keep u in bounds)
  w_x      [10..1000]    typically 100-500 (state bounds)
  lr1      [1e-4..1e-2]  Phase 1 LR, typically 1e-3
  lr2      [1e-5..1e-3]  Phase 2 LR, typically lower than lr1 (refine)

ENGINEERING WISDOM:
  - w_ode should DOMINATE during early Phase 1: physics first, control later
  - w_u and w_x should be MUCH LARGER than tracking weights so constraints
    are enforced strongly when activated in Phase 2
  - lr2 < lr1 (typical 5-10x lower): Phase 2 refines, not overhauls
  - Reference Kardamaki's published-best: w_ode=131.2, w_ic=2.34, w_ytrk=6.28,
    w_utrk=6.73, w_du=32.5, w_u=3243.4, w_x=326.0, lr1=1.0e-3, lr2=2.7e-4

OBJECTIVE: minimize the closed-loop steady-state offset (combined metric of
mean+max abs(y(t_end) - y_sp) over 500 tracking + 500 disturbance episodes).
Lower = better. Kardamaki's published best gives ~0.02 m offset.

You see HISTORY of past trials with their hparams and resulting scores.
Propose ONE config to try next. STRICT JSON only, no markdown, no commentary:
  {"w_ode": <float>, "w_ic": <float>, "w_ytrk": <float>, "w_utrk": <float>,
   "w_du": <float>, "w_u": <float>, "w_x": <float>, "lr1": <float>,
   "lr2": <float>, "reason": "<one short sentence>"}
"""


def _load_dotenv_chain():
    here = pathlib.Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        for name in (".env", "phase_2/.env"):
            cand = parent / name
            if cand.exists() and cand.is_file():
                for ln in cand.read_text().splitlines():
                    if "=" not in ln or ln.strip().startswith("#"):
                        continue
                    k, v = ln.split("=", 1)
                    os.environ.setdefault(k.strip(),
                                            v.strip().strip('"').strip("'"))


_load_dotenv_chain()


class LLMTuner:
    name = "llm"

    def __init__(self, seed: int = 0, warm_start: bool = True,
                 model: str = "claude-sonnet-4-6",
                 temperature: float = 0.4):
        try:
            import anthropic
            self.cli = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"])
        except Exception as e:
            raise RuntimeError(
                f"LLMTuner requires anthropic SDK + ANTHROPIC_API_KEY: {e}")
        self.model = model
        self.temperature = temperature
        self.warm_start = warm_start
        self.seed = seed
        self.history: list = []
        self._warmed = False

    def _format_history(self, max_recent: int = 12) -> str:
        if not self.history:
            return "(no trials yet)"
        lines = []
        for i, h in enumerate(self.history[-max_recent:], 1):
            c = h["cfg"]
            lines.append(
                f"  trial {i:>2}: w_ode={c['w_ode']:.3g} w_ic={c['w_ic']:.3g} "
                f"w_ytrk={c['w_ytrk']:.3g} w_utrk={c['w_utrk']:.3g} "
                f"w_du={c['w_du']:.3g} w_u={c['w_u']:.3g} "
                f"w_x={c['w_x']:.3g} lr1={c['lr1']:.2e} lr2={c['lr2']:.2e} "
                f"-> score={h['score']:.4f}")
        return "\n".join(lines)

    @staticmethod
    def _extract_json(text: str) -> dict:
        for m in reversed(list(re.finditer(r"\{.*?\}", text, re.DOTALL))):
            try:
                obj = json.loads(m.group(0))
            except Exception:
                continue
            if isinstance(obj, dict) and "w_ode" in obj:
                return obj
        return {}

    def ask(self) -> dict:
        # Warm start with Kardamaki's published-best on first call
        if self.warm_start and not self._warmed:
            self._warmed = True
            return _clip(dict(KARDAMAKI_BEST))

        user = (f"HISTORY:\n{self._format_history()}\n\n"
                f"Kardamaki published best score: ~0.02 m offset. "
                f"Best so far this run: "
                f"{min((h['score'] for h in self.history), default=float('inf')):.4f}. "
                f"Propose next config (STRICT JSON only).")
        # Use prompt caching on the system prompt + history block
        resp = self.cli.messages.create(
            model=self.model,
            max_tokens=600,
            temperature=self.temperature,
            system=[{
                "type": "text",
                "text": LLM_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user}],
        )
        raw = "".join(b.text for b in resp.content
                       if getattr(b, "type", None) == "text")
        proposed = self._extract_json(raw)
        if not proposed:
            # Fallback: nudge from Kardamaki best by ±20%
            rng = np.random.default_rng(self.seed + len(self.history))
            proposed = {k: v * float(rng.uniform(0.8, 1.2))
                        for k, v in KARDAMAKI_BEST.items()}
        proposed.pop("reason", None)
        return _clip(proposed)

    def tell(self, cfg: dict, score: float):
        self.history.append({"cfg": cfg, "score": score})


# ---------------------------------------------------------------------------
# Tuner registry
# ---------------------------------------------------------------------------
TUNERS = {
    "random": RandomTuner,
    "bo":     BOTuner,
    "optuna": OptunaTuner,
    "llm":    LLMTuner,
}


def make_tuner(name: str, seed: int = 0):
    if name not in TUNERS:
        raise ValueError(f"Unknown tuner: {name}. Available: {list(TUNERS)}")
    return TUNERS[name](seed=seed)
