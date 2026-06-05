"""LLMAgentOpt - 3-agent + Optuna TPE hyperparameter tuner for Res-Phys NARX.

A physics-aware hyperparameter optimizer that pairs Optuna TPE (the standard
Bayesian-optimization backbone) with a three-agent LLM panel:

    Diagnostic agent (gpt-4o-mini)   - classifies last-trial failure mode
                                            using a physics-aware taxonomy
                                            (extrap_drift, residual_too_strong,
                                             residual_too_weak, physics_dominant,
                                             underfitting, overfitting, ok)
    Strategy agent   (claude opus)   - emits NL rationale + JSON config delta
    Tuning agent     (deterministic) - clamps to search space + enqueues

Optuna TPE drives ~70% of trials; the LLM panel drives the remaining ~30%,
matching Centaur (arXiv:2603.24647) and SLLMBO (arXiv:2410.20302).

Modes (CLI):
  --mode none           pure TPE, no LLM (dev / baseline)
  --mode llambo         LLM warm-start only (3 configs then pure TPE)
  --mode llm_agent_opt  full 3-agent loop (production)   [alias: 'lean3']

Outputs (per study):
  runs/<study_name>/results.json   list of trial dicts (config + metrics + LLM rationale)
  runs/<study_name>/tuner.log      human-readable log
  runs/<study_name>/best.json      best trial + its config

This module was originally named `lean_tuner.py`. It is preserved here under
that filename for backward compatibility with existing run scripts. New code
should import from `llm_agent_opt` (a thin alias that re-exports everything).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

import optuna
from optuna.samplers import TPESampler

from train import (run_trial, build_hparams, PAPER_REF)


# ============================================================================
# Search space - what the tuner is allowed to vary.
# ============================================================================
SEARCH_SPACE = {
    "hidden":         {"type": "categorical",
                          "choices": [(100, 100), (200, 200), (200, 400, 200),
                                       (400, 400, 400), (100, 200, 100),
                                       (200, 400, 400, 200)]},
    "activation":     {"type": "categorical",
                          "choices": ["tanh", "relu", "gelu", "silu"]},
    "lr_adam":        {"type": "loguniform",  "low": 1e-5,  "high": 1e-2},
    "lr_lbfgs":       {"type": "loguniform",  "low": 1e-3,  "high": 1.0},
    "n_epochs_adam":  {"type": "int",         "low": 100,   "high": 2000,
                          "log": False},
    "lbfgs_iters":    {"type": "int",         "low": 100,   "high": 2000,
                          "log": False},
    "batch_size":     {"type": "categorical", "choices": [16, 32, 64, 128]},
    "residual_l2":    {"type": "loguniform",  "low": 1e-8,  "high": 0.1},
    "window":         {"type": "categorical", "choices": [1, 2, 3]},
}


def _filter_to_search_space(cfg: dict) -> dict:
    """Drop keys not in SEARCH_SPACE; drop values outside the declared
    ranges / choices so `study.enqueue_trial` won't raise."""
    out = {}
    for key, val in cfg.items():
        if key not in SEARCH_SPACE:
            continue
        spec = SEARCH_SPACE[key]
        if spec["type"] == "categorical":
            # hidden is sometimes a list - convert to tuple to match choices
            if key == "hidden" and isinstance(val, list):
                val = tuple(val)
            if val not in spec["choices"]:
                continue
            out[key] = val
        elif spec["type"] in ("loguniform", "uniform", "int"):
            lo, hi = spec["low"], spec["high"]
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if not (lo <= v <= hi):
                # clamp instead of dropping
                v = max(lo, min(hi, v))
            if spec["type"] == "int":
                v = int(v)
            out[key] = v
    return out


def sample_from_space(trial: optuna.Trial) -> dict:
    """Map Optuna trial -> ResPhysNARXHparams overrides dict."""
    out = {}
    # Categorical
    out["hidden"]     = tuple(trial.suggest_categorical("hidden",
                                                            SEARCH_SPACE["hidden"]["choices"]))
    out["activation"] = trial.suggest_categorical("activation",
                                                      SEARCH_SPACE["activation"]["choices"])
    out["batch_size"] = trial.suggest_categorical("batch_size",
                                                      SEARCH_SPACE["batch_size"]["choices"])
    out["window"]     = trial.suggest_categorical("window",
                                                      SEARCH_SPACE["window"]["choices"])
    # Numeric
    out["lr_adam"]    = trial.suggest_float("lr_adam",
                                                SEARCH_SPACE["lr_adam"]["low"],
                                                SEARCH_SPACE["lr_adam"]["high"],
                                                log=True)
    out["lr_lbfgs"]   = trial.suggest_float("lr_lbfgs",
                                                SEARCH_SPACE["lr_lbfgs"]["low"],
                                                SEARCH_SPACE["lr_lbfgs"]["high"],
                                                log=True)
    out["n_epochs_adam"] = trial.suggest_int("n_epochs_adam",
                                                  SEARCH_SPACE["n_epochs_adam"]["low"],
                                                  SEARCH_SPACE["n_epochs_adam"]["high"])
    out["lbfgs_iters"]   = trial.suggest_int("lbfgs_iters",
                                                  SEARCH_SPACE["lbfgs_iters"]["low"],
                                                  SEARCH_SPACE["lbfgs_iters"]["high"])
    out["residual_l2"]   = trial.suggest_float("residual_l2",
                                                    SEARCH_SPACE["residual_l2"]["low"],
                                                    SEARCH_SPACE["residual_l2"]["high"],
                                                    log=True)
    return out


# ============================================================================
# LLM clients - lazy-imported so the file works without LLM keys
# ============================================================================
# Hard timeout on every LLM call (default 5 min) so a flaky API call can
# never hang the tuning loop indefinitely. Override with the env var
# LLM_CALL_TIMEOUT_S if you need a longer ceiling.
_LLM_TIMEOUT_S = float(os.environ.get("LLM_CALL_TIMEOUT_S", "300.0"))


def _anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                  timeout=_LLM_TIMEOUT_S)


def _openai_client():
    import openai
    return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                            timeout=_LLM_TIMEOUT_S)


def _claude_call(prompt: str, system: str = "",
                  model: str = "claude-opus-4-7",
                  max_tokens: int = 2000) -> str:
    """Single-shot Claude call."""
    client = _anthropic_client()
    resp = client.messages.create(
        model=model, max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}])
    # Concatenate text blocks
    return "".join(b.text for b in resp.content if b.type == "text")


def _gpt_call(prompt: str, system: str = "",
               model: str = "gpt-4o-mini",
               max_tokens: int = 2000) -> str:
    """Single-shot GPT-4o-mini call."""
    client = _openai_client()
    resp = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": prompt}])
    return resp.choices[0].message.content


# ============================================================================
# Agents
# ============================================================================
DIAGNOSTIC_SYSTEM = """You are the diagnostic agent for a hyperparameter tuner targeting a
Residual-Physics NARX neural network on the Bequette CSTR plant from Thosar et al. 2025.

The model architecture is: y_hat = physics_step(y(t-1), u(t-1)) + NN(window).
Physics is a fixed RK4/LSODA step (correct ODE); the NN only learns model-plant mismatch.

Your job: given the result of one training trial, classify the dominant failure mode
in ONE sentence. Be concrete - reference specific metric values you read.

Possible failure modes:
  - "underfitting" : both Test 1 and Test 2 MAE >> 1e-3, val_loss high
  - "overfitting"  : val_loss << test MAE (gap > 10x)
  - "extrap_drift" : Test 2 MAE >> Test 1 MAE by more than 50x (NN extrapolating badly)
  - "residual_too_strong": residual_l2 likely too high - NN can't correct enough
  - "residual_too_weak":   residual_l2 too low - NN overfits training noise
  - "physics_dominant":    NN residual saturated at zero, physics carries everything
  - "ok": metrics look healthy, just optimize via TPE next
"""

DIAGNOSTIC_USER_TEMPLATE = """Trial config: {hp}
Trial metrics:
  val_loss:              {val_loss:.4e}
  Test 1 one-step MAE:   {mae_t1_one_step:.4e}
  Test 2 one-step MAE:   {mae_t2_one_step:.4e}
  Test 1 autoreg MAE:    {mae_t1_autoreg:.4e}
  Test 2 autoreg MAE:    {mae_t2_autoreg:.4e}
  train time:            {train_time_s:.1f}s

Reference (paper Table 2):
  NARX Test1=0.001508 Test2=0.01934 ; PI-NARX Test1=0.001242 Test2=0.01556

Return a JSON object with two fields:
  "failure_mode": one of {failure_modes}
  "evidence":     one-sentence justification citing specific numbers
"""

STRATEGY_SYSTEM = """You are the strategy agent for a Res-Phys NARX hyperparameter tuner.

Given a diagnostic and the trajectory of recent trials, propose a NEW configuration
that addresses the diagnosis. The tunable hyperparameters are:
  hidden        : tuple of int (architecture)
  activation    : tanh|relu|gelu|silu
  lr_adam       : float in [1e-5, 1e-2] (log scale)
  lr_lbfgs      : float in [1e-3, 1.0]  (log scale)
  n_epochs_adam : int in [100, 2000]
  lbfgs_iters   : int in [100, 2000]
  batch_size    : 16|32|64|128
  residual_l2   : float in [1e-8, 0.1] (log scale)
  window        : 1|2|3

Be SPECIFIC about which knob you move and WHY, drawn from the diagnostic.
Output: a one-paragraph rationale (3-5 sentences) followed by a JSON config object.

Example output:
RATIONALE: The diagnostic flags extrap_drift (Test 2 30x worse than Test 1) and
the NN residual norms are large. Tighten the residual via residual_l2=1e-4
(was 1e-7) so the NN trusts physics more on out-of-distribution inputs.
Also widen the NN to (300, 600, 300) for more capacity near transients.

CONFIG: {"residual_l2": 1e-4, "hidden": [300, 600, 300]}
"""


def diagnostic_agent(result: dict, mock: bool = False) -> dict:
    """Run the diagnostic agent on one trial result. Returns
    {failure_mode: str, evidence: str}."""
    if mock:
        # Trivial heuristic fallback for dev / no-LLM mode
        t1 = result["mae_t1_one_step"]; t2 = result["mae_t2_one_step"]
        if t2 > 50 * t1:
            mode = "extrap_drift"; ev = f"Test2/Test1 ratio = {t2/t1:.1f}"
        elif t1 > 1e-2:
            mode = "underfitting"; ev = f"Test1 MAE={t1:.2e} >> 1e-3"
        elif result["val_loss"] > 10 * t1:
            mode = "overfitting";  ev = f"val/test gap large"
        else:
            mode = "ok";           ev = "metrics within target band"
        return {"failure_mode": mode, "evidence": ev}

    prompt = DIAGNOSTIC_USER_TEMPLATE.format(
        hp=json.dumps(result["hp"]),
        val_loss=result["val_loss"],
        mae_t1_one_step=result["mae_t1_one_step"],
        mae_t2_one_step=result["mae_t2_one_step"],
        mae_t1_autoreg=result["mae_t1_autoreg"],
        mae_t2_autoreg=result["mae_t2_autoreg"],
        train_time_s=result["train_time_s"],
        failure_modes=json.dumps(["underfitting", "overfitting",
                                     "extrap_drift", "residual_too_strong",
                                     "residual_too_weak", "physics_dominant",
                                     "ok"]))
    raw = _gpt_call(prompt, system=DIAGNOSTIC_SYSTEM)
    # Extract first JSON object
    try:
        start = raw.find("{"); end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {"failure_mode": "ok",
                 "evidence": f"diagnostic parse failure; raw={raw[:200]}"}


def strategy_agent(diagnostic: dict, recent_trials: list[dict],
                     last_hp: dict, mock: bool = False) -> dict:
    """Strategy agent returns {rationale: str, config: dict}."""
    if mock:
        # Trivial heuristic mock: nudge the most relevant knob
        mode = diagnostic["failure_mode"]
        cfg = dict(last_hp)
        if mode == "extrap_drift":
            cfg["residual_l2"] = max(1e-4, last_hp.get("residual_l2", 1e-7) * 10)
        elif mode == "underfitting":
            cfg["n_epochs_adam"] = min(2000, int(last_hp.get("n_epochs_adam", 1000) * 1.5))
        elif mode == "overfitting":
            cfg["residual_l2"] = min(0.1, last_hp.get("residual_l2", 1e-7) * 10)
        elif mode == "physics_dominant":
            cfg["residual_l2"] = max(1e-8, last_hp.get("residual_l2", 1e-7) / 10)
        return {"rationale": f"[mock] {mode} -> heuristic nudge",
                 "config": cfg}

    trials_summary = "\n".join(
        f"  Trial {i}: hp={json.dumps({k: t['hp'][k] for k in t['hp'] if k != 'device'})} "
        f"-> objective={t['objective']:.4e} (T1={t['mae_t1_one_step']:.4e}, "
        f"T2={t['mae_t2_one_step']:.4e})"
        for i, t in enumerate(recent_trials[-5:]))
    prompt = (f"Diagnostic: {json.dumps(diagnostic)}\n\n"
                f"Last 5 trials:\n{trials_summary}\n\n"
                f"Propose the next config. Vary at most 2-3 knobs from the last trial.")
    raw = _claude_call(prompt, system=STRATEGY_SYSTEM)
    rationale = raw.split("CONFIG:")[0].replace("RATIONALE:", "").strip()
    try:
        cfg_str = raw.split("CONFIG:")[1].strip()
        start = cfg_str.find("{"); end = cfg_str.rfind("}") + 1
        cfg = json.loads(cfg_str[start:end])
        # Convert hidden list -> tuple
        if "hidden" in cfg and isinstance(cfg["hidden"], list):
            cfg["hidden"] = tuple(cfg["hidden"])
    except Exception:
        cfg = {}
    return {"rationale": rationale, "config": cfg}


def tuning_agent(strategy_config: dict, last_hp: dict) -> dict:
    """Merges strategy's NL-proposed config with the previous trial's hp
    to produce a complete config dict ready for run_trial(). The Tuning
    agent here is purely deterministic / programmatic - no LLM call needed
    once Strategy has emitted JSON."""
    cfg = dict(last_hp)
    cfg.update(strategy_config)
    # Sanity clamps to keep within the search space bounds
    if "lr_adam" in cfg:
        cfg["lr_adam"] = float(np.clip(cfg["lr_adam"], 1e-5, 1e-2))
    if "lr_lbfgs" in cfg:
        cfg["lr_lbfgs"] = float(np.clip(cfg["lr_lbfgs"], 1e-3, 1.0))
    if "n_epochs_adam" in cfg:
        cfg["n_epochs_adam"] = int(np.clip(cfg["n_epochs_adam"], 100, 2000))
    if "lbfgs_iters" in cfg:
        cfg["lbfgs_iters"] = int(np.clip(cfg["lbfgs_iters"], 100, 2000))
    if "residual_l2" in cfg:
        cfg["residual_l2"] = float(np.clip(cfg["residual_l2"], 1e-8, 0.1))
    return cfg


# ============================================================================
# Main tuning loop
# ============================================================================
def tune(study_name: str = "resphys_default",
           n_trials: int = 50,
           mode: str = "none",          # none | llambo | llm_agent_opt | lean3 (alias)
           noise: str | None = None,    # None | "snr35" | "snr100"
           protocol: str = "grid",      # "grid" | "aprbs"
           llm_ratio: float = 0.3,      # fraction of trials driven by LLM
           qf_levels: int = 10, qc_levels: int = 10,
           seed: int = 0,
           device: str = "cuda",
           out_root: str = "runs",
           ) -> dict:
    """Run an LLMAgentOpt tuning study. Returns the best trial dict."""
    # Accept "lean3" as a backward-compat alias for "llm_agent_opt"
    if mode == "lean3":
        mode = "llm_agent_opt"
    out_dir = Path(out_root) / study_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "tuner.log"

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"=== LLMAgentOpt :: study={study_name} mode={mode} n_trials={n_trials} "
        f"noise={noise} protocol={protocol} ===")

    # ----- Sampler: TPE with optional LLM warm-start -----
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(study_name=study_name, sampler=sampler,
                                    direction="minimize")

    results: list[dict] = []
    last_hp = build_hparams({"seed": seed, "device": device}).__dict__

    def objective(trial: optuna.Trial) -> float:
        # If a config was enqueued (by LLM), Optuna fills `trial.params` with it.
        overrides = sample_from_space(trial)
        overrides["seed"]   = seed
        overrides["device"] = device
        try:
            r = run_trial(hp_overrides=overrides,
                            qf_levels=qf_levels, qc_levels=qc_levels,
                            noise=noise, protocol=protocol)
        except Exception as e:
            log(f"  trial failed: {e}\n{traceback.format_exc()[:400]}")
            return 1.0
        r["trial_idx"] = trial.number
        r["source"] = "tpe"
        results.append(r)
        log(f"  trial {trial.number:>3} [tpe ]  "
              f"T1={r['mae_t1_one_step']:.4e}  T2={r['mae_t2_one_step']:.4e}  "
              f"obj={r['objective']:.4e}  ({r['train_time_s']:.0f}s)")
        # Persist incremental
        (out_dir / "results.json").write_text(json.dumps(results, indent=2,
                                                              default=str))
        nonlocal last_hp
        last_hp = r["hp"]
        return r["objective"]

    # ----- LLM warm-start (LLAMBO-style) -----
    if mode in ("llambo", "llm_agent_opt"):
        log(f"  warm-start: requesting 3 LLM-proposed configs")
        # Use the heuristic mocks for warm-start to avoid needing an LLM key
        # just to run the search. If keys are set, switch to real LLM.
        warm_mock = not bool(os.environ.get("ANTHROPIC_API_KEY"))
        baseline_result = {"hp": last_hp, "val_loss": 1e-3,
                              "mae_t1_one_step": 0.001, "mae_t2_one_step": 0.01,
                              "mae_t1_autoreg": 0.005, "mae_t2_autoreg": 0.05,
                              "train_time_s": 30.0}
        for k in range(3):
            diag = diagnostic_agent(baseline_result, mock=warm_mock)
            strat = strategy_agent(diag, results[-5:], last_hp, mock=warm_mock)
            cfg = tuning_agent(strat["config"], last_hp)
            log(f"  warm-start {k}: diag={diag['failure_mode']}  "
                  f"strat=\"{strat['rationale'][:80]}...\"")
            try:
                enq = _filter_to_search_space(cfg)
                if enq:
                    study.enqueue_trial(enq)
                else:
                    log(f"  warm-start {k}: no valid params to enqueue")
            except Exception as e:
                log(f"  warm-start {k}: enqueue failed: {e}")

    # ----- Main optimization loop -----
    # If mode='llm_agent_opt', after every ceil(1/llm_ratio) TPE trials, run the agent loop.
    if mode != "llm_agent_opt":
        study.optimize(objective, n_trials=n_trials)
    else:
        n_done = 0
        interval = max(1, int(round(1.0 / llm_ratio)))
        while n_done < n_trials:
            # Run `interval-1` TPE trials, then 1 LLM-driven trial
            chunk = min(interval - 1, n_trials - n_done)
            if chunk > 0:
                study.optimize(objective, n_trials=chunk)
                n_done += chunk
                if n_done >= n_trials: break
            # LLM agent loop. If anything in here fails or times out, log it
            # and fall back to TPE for this round.
            if results:
                try:
                    diag = diagnostic_agent(results[-1],
                                              mock=not bool(os.environ.get("OPENAI_API_KEY")))
                    strat = strategy_agent(diag, results, last_hp,
                                              mock=not bool(os.environ.get("ANTHROPIC_API_KEY")))
                    cfg = tuning_agent(strat["config"], last_hp)
                    log(f"  agent: diag={diag['failure_mode']}  "
                          f"strat=\"{strat['rationale'][:80]}...\"")
                except Exception as e:
                    log(f"  agent failed ({type(e).__name__}: {str(e)[:120]}); "
                          f"continuing on TPE only")
                    cfg = None
                    strat = {"rationale": f"agent error: {e}", "config": {}}
                if cfg is not None:
                    try:
                        enq = _filter_to_search_space(cfg)
                        if enq:
                            study.enqueue_trial(enq)
                        else:
                            log(f"  agent: no valid params to enqueue")
                    except Exception as e:
                        log(f"  agent enqueue failed: {e}")
            study.optimize(objective, n_trials=1)
            # Mark the most recent trial as LLM-driven for the log
            if results:
                results[-1]["source"] = "llm"
                results[-1]["llm_rationale"] = strat.get("rationale") if 'strat' in dir() else None
                (out_dir / "results.json").write_text(
                    json.dumps(results, indent=2, default=str))
            n_done += 1

    # ----- Final report -----
    if not results:
        log("  no completed trials")
        return {}
    best = min(results, key=lambda r: r["objective"])
    (out_dir / "best.json").write_text(json.dumps(best, indent=2, default=str))
    log(f"=== BEST trial {best['trial_idx']} obj={best['objective']:.4e}")
    log(f"    T1 one-step = {best['mae_t1_one_step']:.4e}  "
        f"(paper PI-NARX = {PAPER_REF['pinarx_t1']:.4e})")
    log(f"    T2 one-step = {best['mae_t2_one_step']:.4e}  "
        f"(paper PI-NARX = {PAPER_REF['pinarx_t2']:.4e})")
    log(f"    Config: {json.dumps(best['hp'], default=str)}")
    return best


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study-name", default="resphys_default")
    ap.add_argument("--n-trials",   type=int, default=50)
    ap.add_argument("--mode",       choices=["none", "llambo",
                                                  "llm_agent_opt", "lean3"],
                    default="none",
                    help="none=pure TPE; llambo=warm-start only; "
                         "llm_agent_opt=full 3-agent loop (recommended). "
                         "'lean3' is kept as a backward-compat alias.")
    ap.add_argument("--noise",      choices=[None, "snr35", "snr100"],
                    default=None)
    ap.add_argument("--protocol",   choices=["grid", "aprbs"], default="grid",
                    help="grid (10x10 open-interval, default) | aprbs "
                         "(paper-exact 5000-min APRBS, 2000/3000 split). "
                         "Use 'aprbs' for the apples-to-apples reviewer-proof baseline.")
    ap.add_argument("--llm-ratio",  type=float, default=0.3)
    ap.add_argument("--qf-levels",  type=int, default=10)
    ap.add_argument("--qc-levels",  type=int, default=10)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available()
                                                else "cpu")
    args = ap.parse_args()
    best = tune(study_name=args.study_name, n_trials=args.n_trials,
                  mode=args.mode, noise=args.noise, protocol=args.protocol,
                  llm_ratio=args.llm_ratio,
                  qf_levels=args.qf_levels, qc_levels=args.qc_levels,
                  seed=args.seed, device=args.device)
