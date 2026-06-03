"""LLM-AutoOpt + Random + BO + Optuna bench for the PC-Gym case studies.

Tunes PINN-MPC loss weights (w_ode, w_ic, w_ytrk, w_utrk, w_du, w_u, w_x, lr1, lr2)
for crystallization, or w_xtrk-extended set for four-tank.

Usage:
  python -m agentic_pcgym.bench --case crystallization --tuners random llm --n-trials 5
  python -m agentic_pcgym.bench --case fourtank        --tuners llm        --n-trials 25 --K1 5000 --K2 5000
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .pinn_crystallization import PINN_Crystallization, CrystPINNHparams, DEVICE
from .pinn_fourtank import PINN_FourTank, FourTankPINNHparams
from .pinn_training import (train_pinn_crystallization, train_pinn_fourtank)
from .data_gen import (sample_crystallization_episodes, sample_fourtank_episodes)
from .evaluator import (evaluate_crystallization, evaluate_fourtank)
from .nmpc_crystallization import CrystallizationNMPC, CrystOperatingPoint
from .nmpc_fourtank import FourTankNMPC, FourTankOperatingPoint
from .plants.crystallization import CrystParams, CV_from_moments, Ln_from_moments
from .plants.fourtank import FourTankParams


# ============================================================================
# Hparam search space (different for each case study)
# ============================================================================
CRYST_HSPACE = {
    "w_ode":   (1.0,   1000.0, True),
    "w_ic":    (0.1,   100.0,  True),
    "w_ytrk":  (0.1,   100.0,  True),
    "w_utrk":  (0.01,  10.0,   True),
    "w_du":    (0.1,   100.0,  True),
    "w_u":     (1.0,   1000.0, True),
    "w_x":     (0.1,   100.0,  True),
    "lr1":     (1e-4,  1e-2,   True),
    "lr2":     (1e-5,  1e-3,   True),
}
CRYST_DEFAULT = {"w_ode": 100.0, "w_ic": 10.0, "w_ytrk": 10.0, "w_utrk": 1.0,
                  "w_du": 1.0, "w_u": 100.0, "w_x": 10.0,
                  "lr1": 1e-3, "lr2": 2e-4}

FT_HSPACE = {
    "w_ode":   (1.0,   1000.0, True),
    "w_ic":    (0.1,   100.0,  True),
    "w_ytrk":  (0.1,   100.0,  True),
    "w_xtrk":  (0.01,  10.0,   True),
    "w_utrk":  (0.01,  10.0,   True),
    "w_du":    (0.1,   100.0,  True),
    "w_u":     (1.0,   1000.0, True),
    "lr1":     (1e-4,  1e-2,   True),
    "lr2":     (1e-5,  1e-3,   True),
}
FT_DEFAULT = {"w_ode": 100.0, "w_ic": 10.0, "w_ytrk": 10.0, "w_xtrk": 1.0,
               "w_utrk": 1.0, "w_du": 1.0, "w_u": 100.0,
               "lr1": 1e-3, "lr2": 2e-4}


def _clip(cfg: dict, hspace: dict) -> dict:
    return {n: max(lo, min(hi, float(cfg.get(n, 0.5*(lo+hi)))))
             for n, (lo, hi, _) in hspace.items()}


# ============================================================================
# Tuners (Random, BO, Optuna, LLM) - simplified versions of agentic_pinn_mpc/tuners.py
# ============================================================================
class RandomTuner:
    name = "random"
    def __init__(self, hspace, seed=0):
        self.hspace = hspace; self.rng = np.random.default_rng(seed)
        self.history = []
    def ask(self):
        cfg = {}
        for n, (lo, hi, log) in self.hspace.items():
            cfg[n] = (math.exp(self.rng.uniform(math.log(lo), math.log(hi)))
                       if log else self.rng.uniform(lo, hi))
        return _clip(cfg, self.hspace)
    def tell(self, cfg, score): self.history.append({"cfg": cfg, "score": score})


class BOTuner:
    name = "bo"
    def __init__(self, hspace, seed=0):
        from skopt import Optimizer
        from skopt.space import Real
        space = []
        for n, (lo, hi, log) in hspace.items():
            space.append(Real(lo, hi,
                                prior="log-uniform" if log else "uniform", name=n))
        self.opt = Optimizer(space, base_estimator="GP", random_state=seed,
                              acq_func="LCB", n_initial_points=3)
        self.names = [s.name for s in space]
        self.hspace = hspace; self.history = []
    def ask(self):
        x = self.opt.ask()
        return _clip({n: v for n, v in zip(self.names, x)}, self.hspace)
    def tell(self, cfg, score):
        self.opt.tell([cfg[n] for n in self.names], score)
        self.history.append({"cfg": cfg, "score": score})


class OptunaTuner:
    name = "optuna"
    def __init__(self, hspace, seed=0):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed))
        self.hspace = hspace
        self._pending = None
        self.history = []
    def ask(self):
        trial = self.study.ask()
        cfg = {}
        for n, (lo, hi, log) in self.hspace.items():
            cfg[n] = trial.suggest_float(n, lo, hi, log=log)
        self._pending = trial
        return _clip(cfg, self.hspace)
    def tell(self, cfg, score):
        self.study.tell(self._pending, score)
        self.history.append({"cfg": cfg, "score": score})


def _load_dotenv_chain():
    import os
    here = Path(__file__).resolve()
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


class LLMTuner:
    name = "llm"
    SYSTEM_PROMPT = """You are a SENIOR APC ENGINEER tuning a Physics-Informed
Neural MPC controller for a chemical process. The PINN is trained with a
composite loss (Kardamaki 2026):
  L = w_ode*L_ode + w_ic*L_IC + w_ytrk*L_ytrk + w_utrk*L_utrk
      + w_du*L_du + w_u*L_u + w_x*L_x  (+ w_xtrk for the four-tank MIMO case)

You see HISTORY of past trials (configs + scores). Lower score = better.
Propose ONE config to try next. STRICT JSON only.
"""
    def __init__(self, hspace, seed=0, defaults=None):
        import re
        _load_dotenv_chain()
        try:
            import anthropic, os
            self.cli = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            self._available = True
        except Exception:
            self._available = False
        self.hspace = hspace; self.defaults = defaults or {}
        self.history = []; self._warmed = False
    def ask(self):
        if not self._warmed and self.defaults:
            self._warmed = True
            return _clip(dict(self.defaults), self.hspace)
        if not self._available:
            # Fall back to random
            rng = np.random.default_rng(len(self.history))
            return _clip({n: math.exp(rng.uniform(math.log(lo), math.log(hi))) if log
                          else rng.uniform(lo, hi)
                          for n, (lo, hi, log) in self.hspace.items()},
                         self.hspace)
        import json, re
        hist_str = "\n".join(
            f"  trial {i}: {json.dumps({k: round(v, 4) for k, v in h['cfg'].items()})}"
            f"  -> score={h['score']:.4f}"
            for i, h in enumerate(self.history[-8:], 1)) or "(no trials yet)"
        bounds_str = ", ".join(f"{n} in [{lo}, {hi}]"
                                  for n, (lo, hi, _) in self.hspace.items())
        user = (f"HISTORY:\n{hist_str}\n\nBOUNDS: {bounds_str}\n"
                "Propose the next config as STRICT JSON (the same keys as above).")
        resp = self.cli.messages.create(
            model="claude-sonnet-4-6", max_tokens=600, temperature=0.4,
            system=[{"type": "text", "text": self.SYSTEM_PROMPT,
                      "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}])
        raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        for m in reversed(list(re.finditer(r"\{.*?\}", raw, re.DOTALL))):
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and any(k in obj for k in self.hspace):
                    return _clip(obj, self.hspace)
            except Exception:
                continue
        # Fallback if JSON parse fails
        return _clip(dict(self.defaults), self.hspace)
    def tell(self, cfg, score):
        self.history.append({"cfg": cfg, "score": score})


TUNERS = {"random": RandomTuner, "bo": BOTuner,
           "optuna": OptunaTuner, "llm": LLMTuner}


# ============================================================================
# Train-and-score wrappers for each case study
# ============================================================================
def train_and_score_crystallization(cfg: dict, episodes: dict,
                                      K1: int, K2: int, bs: int,
                                      n_eval_reps: int = 5) -> dict:
    hp = CrystPINNHparams(K1=K1, K2=K2, bs=bs, **{k: float(cfg[k])
        for k in ["w_ode", "w_ic", "w_ytrk", "w_utrk", "w_du", "w_u", "w_x",
                   "lr1", "lr2"]})
    net = PINN_Crystallization().to(DEVICE)
    t0 = time.time()
    hist = train_pinn_crystallization(net, episodes, hp, verbose=False)
    if hist.get("nan_at") is not None:
        return {"score": 1e6, "train_time": time.time() - t0, "nan": True}
    # Closed-loop eval (use PINN as controller via single forward pass per step)
    @torch.no_grad()
    def pinn_query(mu0, mu1, mu2, mu3, c, cv_sp, ln_sp):
        z = lambda v: torch.tensor([float(v)], device=DEVICE)
        t_step = z(CrystOperatingPoint().__class__.__dict__.get('Ts', 1.0))
        Tc_ic = z(32.0)
        _,_,_,_,_,Tc = net(z(1.0), z(mu0), z(mu1), z(mu2), z(mu3),
                               z(c), z(cv_sp), z(ln_sp), Tc_ic)
        return float(Tc.item())
    metrics = evaluate_crystallization(pinn_query, n_reps=n_eval_reps,
                                          seed=42, verbose=False)
    return {"score": -metrics["median_reward_pi"],
             "optimality_gap": metrics["optimality_gap"],
             "MAD": metrics["MAD"], "train_time": time.time() - t0}


def train_and_score_fourtank(cfg: dict, episodes: dict,
                              K1: int, K2: int, bs: int,
                              n_eval_reps: int = 5) -> dict:
    hp = FourTankPINNHparams(K1=K1, K2=K2, bs=bs, **{k: float(cfg[k])
        for k in ["w_ode", "w_ic", "w_ytrk", "w_xtrk", "w_utrk", "w_du",
                   "w_u", "lr1", "lr2"]})
    net = PINN_FourTank().to(DEVICE)
    t0 = time.time()
    hist = train_pinn_fourtank(net, episodes, hp, verbose=False)
    if hist.get("nan_at") is not None:
        return {"score": 1e6, "train_time": time.time() - t0, "nan": True}
    @torch.no_grad()
    def pinn_query(h1, h2, h3, h4, h1_sp, h2_sp, v1_p, v2_p):
        z = lambda v: torch.tensor([float(v)], device=DEVICE)
        _, _, _, _, v1, v2 = net(z(1.0), z(h1), z(h2), z(h3), z(h4),
                                    z(h1_sp), z(h2_sp), z(v1_p), z(v2_p))
        return (float(v1.item()), float(v2.item()))
    metrics = evaluate_fourtank(pinn_query, n_reps=n_eval_reps, seed=42,
                                  verbose=False)
    return {"score": -metrics["median_reward_pi"],
             "optimality_gap": metrics["optimality_gap"],
             "MAD": metrics["MAD"], "train_time": time.time() - t0}


# ============================================================================
# Bench: run a tuner for n_trials, save results
# ============================================================================
def run_bench(case: str, tuner_name: str, n_trials: int,
                episodes: dict, K1: int, K2: int, bs: int,
                out_dir: Path, n_eval_reps: int = 5, seed: int = 0):
    hspace = CRYST_HSPACE if case == "crystallization" else FT_HSPACE
    default = CRYST_DEFAULT if case == "crystallization" else FT_DEFAULT
    tuner_cls = TUNERS[tuner_name]
    if tuner_name == "llm":
        tuner = tuner_cls(hspace, seed=seed, defaults=default)
    else:
        tuner = tuner_cls(hspace, seed=seed)
    trial_fn = (train_and_score_crystallization if case == "crystallization"
                 else train_and_score_fourtank)
    trials = []; best = float("inf"); best_cfg = None
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_trials + 1):
        cfg = tuner.ask()
        t0 = time.time()
        res = trial_fn(cfg, episodes, K1, K2, bs, n_eval_reps)
        elapsed = time.time() - t0
        score = res["score"]
        tuner.tell(cfg, score)
        if score < best: best, best_cfg = score, dict(cfg)
        trials.append({"iter": i, "cfg": cfg, **res, "elapsed_s": elapsed})
        print(f"  [{tuner_name}] iter {i:>2}/{n_trials}: score={score:.4f}  "
               f"best={best:.4f}  ({elapsed:.0f}s)")
        # Checkpoint after every trial
        with (out_dir / f"{tuner_name}.json").open("w") as f:
            json.dump({"tuner": tuner_name, "case": case,
                        "best": best, "best_cfg": best_cfg,
                        "trials": trials}, f, indent=2, default=str)
    return {"best": best, "best_cfg": best_cfg, "trials": trials}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["crystallization", "fourtank"],
                     required=True)
    ap.add_argument("--tuners", nargs="+", default=["random", "bo", "optuna", "llm"])
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--K1", type=int, default=2000)
    ap.add_argument("--K2", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--n-train-eps", type=int, default=2000,
                     help="Number of pre-sampled training episodes")
    ap.add_argument("--n-eval-reps", type=int, default=5,
                     help="Closed-loop evaluation reps per trial")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/pcgym")
    a = ap.parse_args()

    print(f"=== PC-Gym bench: {a.case} ===")
    print(f"  Device: {DEVICE}")
    print(f"  Tuners: {a.tuners}")
    print(f"  Trials per tuner: {a.n_trials}")
    print(f"  Train scale: K1={a.K1}, K2={a.K2}, bs={a.bs}")
    print(f"  Eval reps: {a.n_eval_reps}")
    print()

    # Pre-sample training episodes ONCE (NMPC queries are expensive)
    print(f"Generating {a.n_train_eps} training episodes...")
    sample_fn = (sample_crystallization_episodes if a.case == "crystallization"
                  else sample_fourtank_episodes)
    episodes = sample_fn(N=a.n_train_eps, seed=a.seed, query_nmpc=False,
                          verbose=True)
    print()

    out_dir = Path(a.out) / a.case
    summary = {"case": a.case, "config": vars(a), "tuners": {}}
    for tuner_name in a.tuners:
        print(f"\n=== Tuner: {tuner_name} ===")
        r = run_bench(a.case, tuner_name, a.n_trials,
                       episodes, a.K1, a.K2, a.bs, out_dir,
                       n_eval_reps=a.n_eval_reps, seed=a.seed)
        summary["tuners"][tuner_name] = {"best": r["best"],
                                            "best_cfg": r["best_cfg"]}
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir.resolve()}/summary.json")
    print(f"\n{'tuner':<10}{'best score':>15}")
    for name, info in summary["tuners"].items():
        print(f"  {name:<10}{info['best']:>15.4f}")


if __name__ == "__main__":
    main()
