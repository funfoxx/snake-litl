from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from snake_env import SnakeEnv

ITERATION = 1
TOTAL_TIMESTEPS = 200_000
EVAL_FREQ = 5_000
EVAL_EPISODES = 50
BENCHMARK_EPISODES = 1_000
SEED = 7

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / f"iteration{ITERATION}"
BEST_MODEL_PATH = ARTIFACT_DIR / "best_model.zip"
SUMMARY_PATH = ARTIFACT_DIR / "summary.json"
BEST_EXPORT_PATH = ROOT / "best" / f"iteration{ITERATION}.zip"


def score_from_stats(avg_length: float, avg_steps: float) -> float:
    return 0.0 if avg_steps <= 0 else (avg_length**2) / avg_steps


def evaluate_model(model: DQN, env: SnakeEnv, episodes: int) -> dict[str, float]:
    snake_lengths = []
    steps = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        snake_lengths.append(info["snake_len"])
        steps.append(info["steps"])

    avg_length = float(np.mean(snake_lengths))
    avg_steps = float(np.mean(steps))
    return {
        "episodes": episodes,
        "avg_length": avg_length,
        "avg_steps": avg_steps,
        "score": score_from_stats(avg_length, avg_steps),
        "max_length": int(np.max(snake_lengths)),
    }


class ScoreEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: SnakeEnv,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_path: Path,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_path = best_model_path
        self.best_score = float("-inf")
        self.best_timestep = 0
        self.evaluations: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        metrics = evaluate_model(self.model, self.eval_env, self.n_eval_episodes)
        metrics["timesteps"] = int(self.num_timesteps)
        self.evaluations.append(metrics)

        if metrics["score"] > self.best_score:
            self.best_score = metrics["score"]
            self.best_timestep = int(self.num_timesteps)
            self.model.save(self.best_model_path)

        print(
            "[eval] "
            f"t={metrics['timesteps']} "
            f"avg_len={metrics['avg_length']:.3f} "
            f"avg_steps={metrics['avg_steps']:.3f} "
            f"score={metrics['score']:.6f} "
            f"best={self.best_score:.6f}"
        )
        return True


def build_model() -> DQN:
    train_env = Monitor(SnakeEnv())
    return DQN(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2_000,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=0,
        seed=SEED,
        device="auto",
    )


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    eval_env = SnakeEnv()
    callback = ScoreEvalCallback(
        eval_env=eval_env,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        best_model_path=BEST_MODEL_PATH,
    )

    model = build_model()
    print(f"training iteration {ITERATION} for {TOTAL_TIMESTEPS} timesteps")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=False)

    if not BEST_MODEL_PATH.exists():
        model.save(BEST_MODEL_PATH)
        callback.best_timestep = TOTAL_TIMESTEPS
        callback.best_score = float("-inf")

    best_model = DQN.load(BEST_MODEL_PATH)
    benchmark_env = SnakeEnv()
    benchmark = evaluate_model(best_model, benchmark_env, BENCHMARK_EPISODES)

    summary = {
        "iteration": ITERATION,
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "eval_freq": EVAL_FREQ,
        "eval_episodes": EVAL_EPISODES,
        "benchmark_episodes": BENCHMARK_EPISODES,
        "best_eval_score": callback.best_score,
        "best_eval_timestep": callback.best_timestep,
        "evaluations": callback.evaluations,
        "benchmark": benchmark,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    shutil.copyfile(BEST_MODEL_PATH, BEST_EXPORT_PATH)

    print()
    print("benchmark results over 1000 games:")
    print(f"avg snake length: {benchmark['avg_length']:.3f}")
    print(f"avg steps: {benchmark['avg_steps']:.3f}")
    print(f"score: {benchmark['score']:.9f}")
    print(f"best eval timestep: {callback.best_timestep}")
    print(f"best eval score: {callback.best_score:.9f}")

    benchmark_env.close()
    eval_env.close()
    model.get_env().close()


if __name__ == "__main__":
    main()
