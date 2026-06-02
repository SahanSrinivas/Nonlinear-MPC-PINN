"""Phase B: Agentic training monitor.

During PINN-MPC training, every N epochs an LLM agent inspects the loss
component trajectory and may intervene:

  Action               Effect
  ------               ------
  CONTINUE             Keep training as-is
  ADJUST_WEIGHT k mul  Multiply weight w_k by `mul` (in 0.1..10)
  SWITCH_TO_PHASE_2    End Phase 1 early (enable constraint terms)
  RESET_OPTIMIZER      Reset Adam moments (escape local minimum)
  ABORT                Stop training (loss diverging, recovery hopeless)

The monitor uses Sonnet 4.6 with prompt caching for cost efficiency.
Each query costs ~$0.007 cached / $0.016 uncached.

This is the novel "agent watches loss curves" contribution. It complements
Phase A (LLM proposes hparams BEFORE training) by adding "LLM monitors
DURING training".

References:
  - Paper-3 design: this work
  - LLM-as-agent in optimization: emerging field (2024-2026)
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .pinn_siso import (DEVICE, PINN_Controller, PINNHparams,
                          generate_collocation_points)


MONITOR_SYSTEM_PROMPT = """You are a SENIOR APC ENGINEER monitoring the training
of a PINN-MPC controller for a nonlinear water-tank system. The training has
two phases:

  Phase 1: minimize L_ode + L_ic + L_ytrk + L_utrk only
  Phase 2: add L_du + L_u + L_x (constraint terms) and refine

You see a snapshot of the loss-component trajectory every N epochs. You decide
whether to intervene based on TYPICAL training pathologies:

  - L_ode plateaus high (>1e-1) by epoch 1000 -> dynamics not learned
  - L_ytrk grows over time -> tracking lost, possibly Phase 1 ended too early
  - Sudden loss spike that doesn't recover -> Adam needs reset or LR too high
  - All components stagnate -> stuck in local minimum, reset or adjust weights
  - Constraint losses dominate in Phase 2 -> constraint weights too high

VALID ACTIONS (one per call):

  CONTINUE                  - Training is healthy, keep going
  ADJUST k mul              - Multiply weight w_<k> by mul (k in
                              {ode, ic, ytrk, utrk, du, u, x}; mul in 0.1..10)
  SWITCH_TO_PHASE_2         - End Phase 1 now (only valid in Phase 1)
  RESET_OPTIMIZER           - Reset Adam moments at current LR
  ABORT                     - Stop training (loss is diverging hopelessly)

Output STRICT JSON only:
  {"action": "<one of the above>",
   "weight_key": "<one of ode/ic/ytrk/utrk/du/u/x or null>",
   "multiplier": <float or null>,
   "reason": "<one short sentence>"}

Examples:
  {"action": "CONTINUE", "weight_key": null, "multiplier": null,
   "reason": "Loss decreasing monotonically across all components."}
  {"action": "ADJUST", "weight_key": "u", "multiplier": 0.5,
   "reason": "Input bound loss too large in Phase 2; lower its weight."}
"""


def _anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


@dataclass
class MonitorAction:
    action: str = "CONTINUE"
    weight_key: str | None = None
    multiplier: float | None = None
    reason: str = ""


@dataclass
class MonitorState:
    """State that the monitor manipulates between checks."""
    current_phase: int = 1     # 1 or 2
    weights: dict = field(default_factory=dict)
    aborted: bool = False
    switch_to_phase_2_requested: bool = False
    reset_optimizer_requested: bool = False
    actions_log: list = field(default_factory=list)


class TrainingMonitor:
    """LLM agent that watches training and intervenes."""

    def __init__(self, check_every: int = 500,
                  model: str = "claude-sonnet-4-6",
                  enabled: bool = True):
        self.check_every = check_every
        self.model = model
        self.enabled = enabled
        if enabled:
            self.cli = _anthropic_client()
        self.cost_estimate = 0.0  # rough $ tracker

    @staticmethod
    def _extract_action(text: str) -> dict:
        for m in reversed(list(re.finditer(r"\{.*?\}", text, re.DOTALL))):
            try:
                obj = json.loads(m.group(0))
            except Exception:
                continue
            if isinstance(obj, dict) and "action" in obj:
                return obj
        return {"action": "CONTINUE"}

    def query(self, phase: int, epoch: int, total_epochs: int,
                weights: dict, loss_history: list,
                component_history: dict) -> MonitorAction:
        """Ask the LLM for an action given current training state.

        loss_history: list of recent total loss values
        component_history: dict of name -> list of recent values for each loss component
        """
        if not self.enabled:
            return MonitorAction()
        # Build a compact snapshot
        recent = loss_history[-50:] if len(loss_history) > 50 else loss_history
        comp_summary = {}
        for k, v in component_history.items():
            if v:
                comp_summary[k] = {
                    "first": float(v[0]),
                    "mid": float(v[len(v) // 2]) if len(v) > 1 else float(v[0]),
                    "last": float(v[-1]),
                    "min": float(min(v)),
                    "max": float(max(v)),
                }
        user = (
            f"Phase {phase} of 2, epoch {epoch}/{total_epochs}.\n"
            f"Current weights: {json.dumps(weights, default=str)}\n"
            f"Total loss (last {len(recent)} values, oldest first): "
            f"{[round(float(x), 4) for x in recent[:5] + ['...'] + recent[-5:]]}\n"
            f"Component summary: {json.dumps(comp_summary, indent=None)}\n\n"
            f"Propose your action as STRICT JSON."
        )
        try:
            resp = self.cli.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0.0,
                system=[{
                    "type": "text",
                    "text": MONITOR_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user}],
            )
            self.cost_estimate += 0.007  # cached cost
        except Exception as e:
            return MonitorAction(reason=f"LLM call failed: {e}")
        raw = "".join(b.text for b in resp.content
                       if getattr(b, "type", None) == "text")
        a = self._extract_action(raw)
        return MonitorAction(
            action=a.get("action", "CONTINUE"),
            weight_key=a.get("weight_key"),
            multiplier=a.get("multiplier"),
            reason=str(a.get("reason", ""))[:120],
        )


def train_pinn_with_monitor(
    hp: PINNHparams,
    x0_all, u0_all, ysp_all, d0_all,
    monitor: TrainingMonitor,
    hidden_layers: list[int] = [64, 16, 16],
    verbose: bool = False,
    seed: int = 0,
) -> tuple[PINN_Controller, dict]:
    """Train PINN with an LLM monitor allowed to intervene every N epochs."""
    torch.manual_seed(seed)
    model = PINN_Controller(hidden_layers=hidden_layers).to(DEVICE)
    t_wp = torch.cat([torch.arange(0, hp.T_horizon, hp.Ts, device=DEVICE),
                       torch.tensor([hp.T_horizon], device=DEVICE)])
    N_WP = t_wp.shape[0]
    t_col = generate_collocation_points(
        horizon=hp.T_horizon, dt_dense=0.01, dt_medium=0.1, dt_sparse=1.0)
    N_COL = t_col.shape[0]

    weights = dict(w_ode=hp.w_ode, w_ic=hp.w_ic, w_ytrk=hp.w_ytrk,
                    w_utrk=hp.w_utrk, w_du=hp.w_du, w_u=hp.w_u, w_x=hp.w_x)
    state = MonitorState(current_phase=1, weights=dict(weights))

    hist_p1, hist_p2 = [], []
    comp_p1 = {k: [] for k in ["ode", "ytrk", "utrk", "du", "u", "x", "ic"]}
    comp_p2 = {k: [] for k in comp_p1}
    actions_log = []

    def maybe_query_monitor(phase: int, ep: int, total: int,
                              loss_hist: list, comp_hist: dict):
        if ep % monitor.check_every != 0:
            return
        a = monitor.query(phase, ep, total, dict(weights), loss_hist,
                            comp_hist)
        actions_log.append({"phase": phase, "epoch": ep, "action": a.action,
                              "weight_key": a.weight_key,
                              "multiplier": a.multiplier,
                              "reason": a.reason})
        if a.action == "CONTINUE":
            return
        if a.action == "ABORT":
            state.aborted = True
            return
        if a.action == "RESET_OPTIMIZER":
            state.reset_optimizer_requested = True
            return
        if a.action == "SWITCH_TO_PHASE_2":
            state.switch_to_phase_2_requested = True
            return
        if a.action == "ADJUST" and a.weight_key:
            k = a.weight_key
            wk = f"w_{k}"
            if wk in weights and a.multiplier is not None:
                mul = float(a.multiplier)
                mul = max(0.1, min(10.0, mul))
                weights[wk] = float(weights[wk]) * mul

    # ----- Phase 1 -----
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr1)
    for ep in range(1, hp.K1 + 1):
        start = (ep - 1) * hp.bs
        end = start + hp.bs
        x0 = x0_all[start:end]; u0 = u0_all[start:end]
        ysp = ysp_all[start:end]; d0 = d0_all[start:end]
        loss, comps = model.loss(hp.bs, t_wp, N_WP, t_col, N_COL,
                                   x0, u0, ysp, d0,
                                   weights["w_ode"], weights["w_ic"],
                                   weights["w_ytrk"], weights["w_utrk"],
                                   0.0, 0.0, 0.0)
        if torch.isnan(loss):
            actions_log.append({"phase": 1, "epoch": ep,
                                 "action": "NAN_DETECTED"})
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist_p1.append(loss.item())
        l_ode, l_ytrk, l_utrk, l_du, l_u, l_x, l_ic = [c.item() for c in comps]
        comp_p1["ode"].append(l_ode); comp_p1["ytrk"].append(l_ytrk)
        comp_p1["utrk"].append(l_utrk); comp_p1["du"].append(l_du)
        comp_p1["u"].append(l_u); comp_p1["x"].append(l_x)
        comp_p1["ic"].append(l_ic)
        maybe_query_monitor(1, ep, hp.K1, hist_p1, comp_p1)
        if state.aborted:
            return model, {"hist_p1": hist_p1, "hist_p2": hist_p2,
                            "comp_p1": comp_p1, "comp_p2": comp_p2,
                            "actions": actions_log, "aborted": True}
        if state.reset_optimizer_requested:
            opt = torch.optim.Adam(model.parameters(), lr=hp.lr1)
            state.reset_optimizer_requested = False
        if state.switch_to_phase_2_requested:
            break

    # ----- Phase 2 -----
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr2)
    for ep in range(1, hp.K2 + 1):
        start = (ep - 1) * hp.bs
        end = start + hp.bs
        x0 = x0_all[start:end]; u0 = u0_all[start:end]
        ysp = ysp_all[start:end]; d0 = d0_all[start:end]
        loss, comps = model.loss(hp.bs, t_wp, N_WP, t_col, N_COL,
                                   x0, u0, ysp, d0,
                                   weights["w_ode"], weights["w_ic"],
                                   weights["w_ytrk"], weights["w_utrk"],
                                   weights["w_du"], weights["w_u"],
                                   weights["w_x"])
        if torch.isnan(loss):
            actions_log.append({"phase": 2, "epoch": ep,
                                 "action": "NAN_DETECTED"})
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist_p2.append(loss.item())
        l_ode, l_ytrk, l_utrk, l_du, l_u, l_x, l_ic = [c.item() for c in comps]
        comp_p2["ode"].append(l_ode); comp_p2["ytrk"].append(l_ytrk)
        comp_p2["utrk"].append(l_utrk); comp_p2["du"].append(l_du)
        comp_p2["u"].append(l_u); comp_p2["x"].append(l_x)
        comp_p2["ic"].append(l_ic)
        maybe_query_monitor(2, ep, hp.K2, hist_p2, comp_p2)
        if state.aborted:
            break
        if state.reset_optimizer_requested:
            opt = torch.optim.Adam(model.parameters(), lr=hp.lr2)
            state.reset_optimizer_requested = False

    return model, {"hist_p1": hist_p1, "hist_p2": hist_p2,
                    "comp_p1": comp_p1, "comp_p2": comp_p2,
                    "actions": actions_log,
                    "aborted": state.aborted,
                    "monitor_cost_estimate_usd": monitor.cost_estimate,
                    "final_weights": dict(weights)}
