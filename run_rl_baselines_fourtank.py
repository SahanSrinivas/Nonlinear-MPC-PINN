"""run_rl_baselines_fourtank.py - Train SAC/DDPG/PPO and eval with OUR harness.

Why this exists:
  Bloor 2025 reports four-tank optimality gaps for SAC/DDPG/PPO. The arxiv v2
  text quotes "SAC ... 0.579" while the journal Table 5 shows "SAC 0.0537".
  These can't both be right under the same metric. We sidestep the question
  by training the same RL algorithms ourselves and evaluating them through
  evaluate_fourtank() — the IDENTICAL harness used for our PINN-MPC results.
  Apples-to-apples.

Environment:
  Wraps the four-tank plant + Bloor reward (Eq 13) in a Gymnasium env. State
  is (h1..h4, h1_sp, h2_sp, v1_prev, v2_prev) = 8D. Action is (v1, v2) in
  [0, 15] V. Episode = 60 steps × 16.667 s = 1000 s, matching Bloor §4.1.4.

Usage on Colab:
  !pip install stable_baselines3
  !python -u run_rl_baselines_fourtank.py \\
      --algos SAC DDPG PPO --total-timesteps 200000 --n-eval-reps 30 \\
      --out-dir results/rl_baselines_fourtank 2>&1 | tee rl_baselines.log

Estimated runtime on Colab T4:
  - SAC:  ~45 min
  - DDPG: ~45 min
  - PPO:  ~30 min
  - Eval (30 reps × 3 algos): ~5 min total
  - Total: ~2-2.5 hr
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

# stable_baselines3 + gymnasium (installed via pip)
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise

from agentic_pcgym.plants.fourtank import (
    FourTankParams, step as fourtank_step, FourTankScenario)
from agentic_pcgym.nmpc_fourtank import (
    FourTankOperatingPoint, FourTankBounds)
from agentic_pcgym.evaluator import (
    evaluate_fourtank, normalised_reward_fourtank)


class FourTankEnv(gym.Env):
    """Gymnasium wrapper for the four-tank PC-Gym environment.

    Observation (8D): h1, h2, h3, h4, h1_sp, h2_sp, v1_prev, v2_prev
    Action      (2D): v1, v2  in [v_min, v_max]
    Reward         : Bloor Eq 13 (normalised tracking error on h1,h2)
    Episode        : 60 steps (1000 s wall time, dt=16.667 s)
    Reset          : x0 ~ Uniform(0.8, 1.2) * (op.h_*) ;
                     h1_sp ~ U(0.2, 0.7); h2_sp ~ U(0.2, 0.5)
                     (identical to evaluate_fourtank's setpoint sampling)
    """
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.op = FourTankOperatingPoint()
        self.bounds = FourTankBounds()
        self.scen = FourTankScenario()
        self.params = FourTankParams()

        v_lo = float(self.bounds.v_min)
        v_hi = float(self.bounds.v_max)
        h_lo = float(self.bounds.h_min)
        h_hi = float(self.bounds.h_max)

        self.action_space = spaces.Box(
            low=np.array([v_lo, v_lo], dtype=np.float32),
            high=np.array([v_hi, v_hi], dtype=np.float32),
            dtype=np.float32)

        obs_low = np.array(
            [h_lo, h_lo, h_lo, h_lo, h_lo, h_lo, v_lo, v_lo],
            dtype=np.float32)
        obs_high = np.array(
            [h_hi, h_hi, h_hi, h_hi, h_hi, h_hi, v_hi, v_hi],
            dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high,
                                             dtype=np.float32)

        self.rng = np.random.default_rng(seed)
        self._reset_state()

    def _reset_state(self):
        self.x = np.array([self.op.h_1_0, self.op.h_2_0,
                            self.op.h_3_0, self.op.h_4_0], dtype=float)
        self.v_prev = (5.0, 5.0)
        self.h1_sp = float(self.op.h_1_sp)
        self.h2_sp = float(self.op.h_2_sp)
        self.step_count = 0

    def _obs(self) -> np.ndarray:
        return np.array(
            [self.x[0], self.x[1], self.x[2], self.x[3],
             self.h1_sp, self.h2_sp, self.v_prev[0], self.v_prev[1]],
            dtype=np.float32)

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        x0 = np.array([self.op.h_1_0, self.op.h_2_0,
                        self.op.h_3_0, self.op.h_4_0], dtype=float)
        self.x = x0 * self.rng.uniform(0.8, 1.2, 4)
        self.h1_sp = float(self.rng.uniform(0.2, 0.7))
        self.h2_sp = float(self.rng.uniform(0.2, 0.5))
        self.v_prev = (5.0, 5.0)
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        u = np.asarray(action, dtype=float)
        u = np.clip(u, self.bounds.v_min, self.bounds.v_max)
        self.x = fourtank_step(self.x, u, self.scen.dt_s, self.params)
        r = normalised_reward_fourtank(
            self.x[0], self.x[1], u[0], u[1],
            self.v_prev[0], self.v_prev[1],
            self.h1_sp, self.h2_sp, self.bounds)
        self.v_prev = (float(u[0]), float(u[1]))
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.scen.n_steps
        return self._obs(), float(r), terminated, truncated, {}


def policy_to_query(model):
    """Wrap a trained SB3 model as a controller_query function for evaluate_fourtank."""
    def query(h1, h2, h3, h4, h1_sp, h2_sp, v1_prev, v2_prev):
        obs = np.array([h1, h2, h3, h4, h1_sp, h2_sp, v1_prev, v2_prev],
                        dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        return (float(action[0]), float(action[1]))
    return query


def make_model(algo_name: str, env: gym.Env, seed: int = 0):
    if algo_name == "SAC":
        return SAC("MlpPolicy", env, verbose=0, seed=seed,
                    learning_rate=3e-4, buffer_size=100_000,
                    batch_size=256, gamma=0.99, tau=0.005)
    if algo_name == "DDPG":
        n_actions = 2
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions))
        return DDPG("MlpPolicy", env, action_noise=action_noise,
                     verbose=0, seed=seed,
                     learning_rate=1e-3, buffer_size=100_000,
                     batch_size=256, gamma=0.99, tau=0.005)
    if algo_name == "PPO":
        return PPO("MlpPolicy", env, verbose=0, seed=seed,
                    learning_rate=3e-4, n_steps=2048, batch_size=64,
                    n_epochs=10, gamma=0.99, gae_lambda=0.95)
    raise ValueError(f"Unknown algo: {algo_name}")


def train_and_eval(algo_name: str, total_timesteps: int, n_eval_reps: int,
                    out_dir: str, seed: int = 0) -> dict:
    print(f"\n=== Training {algo_name} on four-tank "
          f"(timesteps={total_timesteps:,}) ===")
    env = FourTankEnv(seed=seed)
    env.reset(seed=seed)
    model = make_model(algo_name, env, seed=seed)

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    train_time = time.time() - t0
    print(f"  trained in {train_time:.1f}s ({train_time/60:.1f} min)")

    save_path = os.path.join(out_dir, f"{algo_name.lower()}_fourtank.zip")
    model.save(save_path)
    print(f"  saved model to {save_path}")

    print(f"  Evaluating on {n_eval_reps} closed-loop reps...")
    query = policy_to_query(model)
    t0 = time.time()
    metrics = evaluate_fourtank(query, n_reps=n_eval_reps, seed=42,
                                 verbose=False)
    eval_time = time.time() - t0
    print(f"  evaluated in {eval_time:.1f}s")
    print(f"  median_reward_pi:     {metrics['median_reward_pi']:>10.4f}")
    print(f"  median_reward_oracle: {metrics['median_reward_oracle']:>10.4f}")
    print(f"  optimality_gap:       {metrics['optimality_gap']:>10.4f}")
    print(f"  MAD:                  {metrics['MAD']:>10.4f}")

    return {
        "algo": algo_name,
        "total_timesteps": total_timesteps,
        "train_time_s": float(train_time),
        "eval_time_s": float(eval_time),
        "median_reward_pi": float(metrics["median_reward_pi"]),
        "median_reward_oracle": float(metrics["median_reward_oracle"]),
        "optimality_gap": float(metrics["optimality_gap"]),
        "MAD": float(metrics["MAD"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["SAC", "DDPG", "PPO"],
                     choices=["SAC", "DDPG", "PPO"],
                     help="Which RL algorithms to train")
    ap.add_argument("--total-timesteps", type=int, default=200_000,
                     help="SB3 .learn(total_timesteps=...) per algorithm")
    ap.add_argument("--n-eval-reps", type=int, default=30,
                     help="Reps for evaluate_fourtank() per algorithm")
    ap.add_argument("--out-dir", default="results/rl_baselines_fourtank")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)

    results = {}
    for algo in a.algos:
        results[algo] = train_and_eval(
            algo, a.total_timesteps, a.n_eval_reps,
            a.out_dir, a.seed)

    print("\n" + "=" * 78)
    print("=== Four-tank RL baselines (trained + evaluated on OUR harness) ===")
    print("=" * 78)
    print(f"{'algo':<10}{'median R':>12}{'opt gap':>14}{'MAD':>12}"
          f"{'train (min)':>14}")
    print("-" * 78)
    for algo, r in results.items():
        print(f"{algo:<10}{r['median_reward_pi']:>12.4f}"
              f"{r['optimality_gap']:>14.4f}{r['MAD']:>12.4f}"
              f"{r['train_time_s']/60:>14.1f}")

    print("\n=== Comparison with our PINN-MPC results (from paper figures) ===")
    print(f"{'method':<20}{'opt gap':>14}{'MAD':>12}")
    print("-" * 46)
    for algo, r in results.items():
        print(f"{algo+' (us)':<20}{r['optimality_gap']:>14.4f}"
              f"{r['MAD']:>12.4f}")
    print(f"{'LLM+DPC PINN':<20}{'0.2061':>14}{'0.1567':>12}  <-- our method")
    print(f"{'LLM PINN':<20}{'0.5365':>14}{'0.3719':>12}")
    print(f"{'Bloor untuned PINN':<20}{'1.4557':>14}{'0.6941':>12}")

    out_path = os.path.join(a.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved JSON to {out_path}")


if __name__ == "__main__":
    main()
