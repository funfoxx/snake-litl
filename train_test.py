from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

XDG_CACHE_HOME = Path(".cache")
XDG_CACHE_HOME.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME.resolve()))

MPLCONFIGDIR = Path(".mplconfig")
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from snake_env import SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--selection-episodes", type=int, default=200)
    parser.add_argument("--benchmark-episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--exploration-fraction", type=float, default=0.25)
    parser.add_argument("--skip-best-copy", action="store_true")
    return parser.parse_args()


def make_env(seed: int) -> Monitor:
    env = Monitor(SnakeEnv())
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def build_model(
    env: Monitor,
    seed: int,
    learning_rate: float,
    exploration_fraction: float,
) -> DQN:
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2_000,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        max_grad_norm=10.0,
        policy_kwargs={"net_arch": [256, 256, 256, 128]},
        verbose=1,
        seed=seed,
        device="auto",
    )


def evaluate_model(model: DQN, episodes: int, seed: int) -> dict[str, float | int]:
    env = SnakeEnv()
    lengths = []
    steps = []
    returns = []

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward

        lengths.append(int(info["snake_len"]))
        steps.append(int(info["steps"]))
        returns.append(float(episode_return))

    env.close()

    lengths_arr = np.asarray(lengths, dtype=np.float32)
    steps_arr = np.asarray(steps, dtype=np.float32)
    returns_arr = np.asarray(returns, dtype=np.float32)
    score = float(lengths_arr.mean() / max(1.0, steps_arr.mean()))

    return {
        "episodes": episodes,
        "avg_snake_length": float(lengths_arr.mean()),
        "avg_steps": float(steps_arr.mean()),
        "avg_return": float(returns_arr.mean()),
        "median_snake_length": float(np.median(lengths_arr)),
        "median_steps": float(np.median(steps_arr)),
        "max_snake_length": int(lengths_arr.max()),
        "min_steps": int(steps_arr.min()),
        "score": score,
    }


def metrics_sort_key(metrics: dict[str, float | int]) -> tuple[float, float, float, float]:
    return (
        float(metrics["avg_snake_length"]),
        -float(metrics["avg_steps"]),
        float(metrics["score"]),
        float(metrics["avg_return"]),
    )


class PeriodicCheckpointCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        eval_freq: int,
        eval_episodes: int,
        eval_seed: int,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.evaluations_path = self.run_dir / "evaluations.json"
        self.evaluations: list[dict[str, object]] = []

    def _on_training_start(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        checkpoint_path = self.checkpoint_dir / f"step_{self.num_timesteps}.zip"
        self.model.save(checkpoint_path)
        metrics = evaluate_model(
            self.model,
            episodes=self.eval_episodes,
            seed=self.eval_seed,
        )
        record: dict[str, object] = {
            "timesteps": self.num_timesteps,
            "checkpoint_path": str(checkpoint_path),
            "metrics": metrics,
        }
        self.evaluations.append(record)
        self.evaluations_path.write_text(json.dumps(self.evaluations, indent=2))
        print(
            "checkpoint",
            self.num_timesteps,
            "avg_len",
            f"{metrics['avg_snake_length']:.3f}",
            "avg_steps",
            f"{metrics['avg_steps']:.3f}",
            "score",
            f"{metrics['score']:.6f}",
        )
        return True


def checkpoint_timesteps(path: Path, fallback_timesteps: int) -> int:
    if path.stem.startswith("step_"):
        return int(path.stem.split("_", maxsplit=1)[1])
    return fallback_timesteps


def select_best_checkpoint(
    run_dir: Path,
    final_model_path: Path,
    selection_episodes: int,
    selection_seed: int,
    final_timesteps: int,
) -> dict[str, object]:
    checkpoint_dir = run_dir / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("step_*.zip"), key=lambda path: checkpoint_timesteps(path, final_timesteps))
    if final_model_path not in candidates:
        candidates.append(final_model_path)

    records: list[dict[str, object]] = []
    best_path: Path | None = None
    best_metrics: dict[str, float | int] | None = None

    for candidate in candidates:
        model = DQN.load(candidate)
        metrics = evaluate_model(model, episodes=selection_episodes, seed=selection_seed)
        record: dict[str, object] = {
            "timesteps": checkpoint_timesteps(candidate, final_timesteps),
            "checkpoint_path": str(candidate),
            "metrics": metrics,
        }
        records.append(record)
        if best_metrics is None or metrics_sort_key(metrics) > metrics_sort_key(best_metrics):
            best_path = candidate
            best_metrics = metrics

    assert best_path is not None
    assert best_metrics is not None

    selected_path = run_dir / "best_model.zip"
    shutil.copy2(best_path, selected_path)

    selection_summary = {
        "episodes": selection_episodes,
        "seed": selection_seed,
        "records": records,
        "best_checkpoint_path": str(best_path),
        "best_model_path": str(selected_path),
        "best_timesteps": checkpoint_timesteps(best_path, final_timesteps),
        "best_metrics": best_metrics,
    }

    (run_dir / "selection_metrics.json").write_text(json.dumps(selection_summary, indent=2))
    return selection_summary


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or Path("runs") / f"iteration{args.iteration}"
    run_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)

    env = make_env(args.seed)
    model = build_model(
        env,
        args.seed,
        learning_rate=args.learning_rate,
        exploration_fraction=args.exploration_fraction,
    )

    checkpoint_callback = PeriodicCheckpointCallback(
        run_dir=run_dir,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        eval_seed=args.seed + 10_000,
    )

    start_time = time.time()
    print(f"training DQN for {args.timesteps} timesteps")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    training_time_seconds = time.time() - start_time

    final_model_path = run_dir / "final_model.zip"
    model.save(final_model_path)

    selection = select_best_checkpoint(
        run_dir=run_dir,
        final_model_path=final_model_path,
        selection_episodes=args.selection_episodes,
        selection_seed=args.seed + 15_000,
        final_timesteps=args.timesteps,
    )
    best_model_path = Path(selection["best_model_path"])

    best_model = DQN.load(best_model_path)
    benchmark = evaluate_model(
        best_model,
        episodes=args.benchmark_episodes,
        seed=args.seed + 20_000,
    )

    metrics = {
        "iteration": args.iteration,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "selection_episodes": args.selection_episodes,
        "benchmark_episodes": args.benchmark_episodes,
        "learning_rate": args.learning_rate,
        "exploration_fraction": args.exploration_fraction,
        "training_time_seconds": training_time_seconds,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "periodic_evaluations": checkpoint_callback.evaluations,
        "checkpoint_selection": selection,
        "benchmark": benchmark,
    }

    metrics_path = run_dir / "final_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("benchmark results over", args.benchmark_episodes, "games:")
    print(json.dumps(benchmark, indent=2))
    print("saved metrics to", metrics_path)
    if not args.skip_best_copy:
        best_dir = Path("best")
        best_dir.mkdir(exist_ok=True)
        best_iteration_path = best_dir / f"iteration{args.iteration}.zip"
        shutil.copy2(best_model_path, best_iteration_path)
        print("saved best model to", best_iteration_path)

    env.close()


if __name__ == "__main__":
    main()
