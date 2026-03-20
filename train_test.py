from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from snake_env import SnakeEnv

ROOT = Path(__file__).resolve().parent


@dataclass(slots=True)
class RunConfig:
    iteration: int = 2
    run_name: str = "main"
    total_timesteps: int = 300_000
    eval_freq: int = 10_000
    eval_episodes: int = 50
    benchmark_episodes: int = 1_000
    seed: int = 7
    learning_rate: float = 1e-4
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    batch_size: int = 256
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 4_000
    exploration_fraction: float = 0.30
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.02
    net_arch: tuple[int, ...] = (256, 256, 128)
    export_best: bool = False


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=2)
    parser.add_argument("--run-name", type=str, default="main")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--benchmark-episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=4_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.30)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.02)
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=[256, 256, 128],
    )
    parser.add_argument("--export-best", action="store_true")
    args = parser.parse_args()

    return RunConfig(
        iteration=args.iteration,
        run_name=args.run_name,
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        benchmark_episodes=args.benchmark_episodes,
        seed=args.seed,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        net_arch=tuple(args.net_arch),
        export_best=args.export_best,
    )


def iteration_artifact_dir(config: RunConfig) -> Path:
    base_dir = ROOT / "artifacts" / f"iteration{config.iteration}"
    if config.run_name == "main":
        return base_dir
    return base_dir / config.run_name


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


def build_model(config: RunConfig) -> DQN:
    train_env = Monitor(SnakeEnv())
    return DQN(
        "MlpPolicy",
        train_env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_initial_eps=config.exploration_initial_eps,
        exploration_final_eps=config.exploration_final_eps,
        policy_kwargs={"net_arch": list(config.net_arch)},
        verbose=0,
        seed=config.seed,
        device="auto",
    )


def main():
    config = parse_args()
    artifact_dir = iteration_artifact_dir(config)
    best_model_path = artifact_dir / "best_model.zip"
    summary_path = artifact_dir / "summary.json"
    best_export_path = ROOT / "best" / f"iteration{config.iteration}.zip"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    if best_model_path.exists():
        best_model_path.unlink()

    eval_env = SnakeEnv()
    callback = ScoreEvalCallback(
        eval_env=eval_env,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.eval_episodes,
        best_model_path=best_model_path,
    )

    model = build_model(config)
    print(
        f"training iteration {config.iteration} "
        f"run={config.run_name} "
        f"for {config.total_timesteps} timesteps"
    )
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback,
        progress_bar=False,
    )

    if not best_model_path.exists():
        model.save(best_model_path)
        callback.best_timestep = config.total_timesteps

    best_model = DQN.load(best_model_path)
    benchmark_env = SnakeEnv()
    benchmark = evaluate_model(best_model, benchmark_env, config.benchmark_episodes)

    summary = {
        "config": asdict(config),
        "best_eval_score": callback.best_score,
        "best_eval_timestep": callback.best_timestep,
        "evaluations": callback.evaluations,
        "benchmark": benchmark,
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if config.export_best:
        shutil.copyfile(best_model_path, best_export_path)

    print()
    print(f"benchmark results over {config.benchmark_episodes} games:")
    print(f"avg snake length: {benchmark['avg_length']:.3f}")
    print(f"avg steps: {benchmark['avg_steps']:.3f}")
    print(f"score: {benchmark['score']:.9f}")
    print(f"best eval timestep: {callback.best_timestep}")
    print(f"best eval score: {callback.best_score:.9f}")
    if config.export_best:
        print(f"exported model: {best_export_path}")

    benchmark_env.close()
    eval_env.close()
    model.get_env().close()


if __name__ == "__main__":
    main()
