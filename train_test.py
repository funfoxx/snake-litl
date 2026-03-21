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
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from snake_env import SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--benchmark-episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--skip-best-copy", action="store_true")
    return parser.parse_args()


def make_env(seed: int) -> Monitor:
    env = Monitor(SnakeEnv())
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def build_model(env: Monitor, seed: int) -> DQN:
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2_000,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        max_grad_norm=10.0,
        policy_kwargs={"net_arch": [256, 256, 128]},
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


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or Path("runs") / f"iteration{args.iteration}"
    run_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)

    env = make_env(args.seed)
    eval_env = make_env(args.seed + 10_000)
    model = build_model(env, args.seed)

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=args.eval_episodes,
        eval_freq=args.eval_freq,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        deterministic=True,
        render=False,
        verbose=1,
    )

    start_time = time.time()
    print(f"training DQN for {args.timesteps} timesteps")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    training_time_seconds = time.time() - start_time

    final_model_path = run_dir / "final_model.zip"
    model.save(final_model_path)

    best_model_path = run_dir / "best_model.zip"
    if not best_model_path.exists():
        best_model_path = final_model_path

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
        "benchmark_episodes": args.benchmark_episodes,
        "training_time_seconds": training_time_seconds,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
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

    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
