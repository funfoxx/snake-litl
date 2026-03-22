import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch as th

os.environ.setdefault("MPLCONFIGDIR", str((Path("artifacts") / "mplconfig").resolve()))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from snake_env import SnakeEnv


class EgocentricExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=192):
        super().__init__(observation_space, features_dim)

        board_space = observation_space["board"]
        feature_space = observation_space["features"]

        self.board_net = th.nn.Sequential(
            th.nn.Conv2d(board_space.shape[0], 32, kernel_size=3, padding=1),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(board_space.sample()[None]).float()
            board_dim = self.board_net(sample).shape[1]

        self.feature_net = th.nn.Sequential(
            th.nn.Linear(feature_space.shape[0], 64),
            th.nn.ReLU(),
        )

        self.combined = th.nn.Sequential(
            th.nn.Linear(board_dim + 64, features_dim),
            th.nn.ReLU(),
        )

    def forward(self, observations):
        board_tensor = observations["board"]
        feature_tensor = observations["features"]
        board_latent = self.board_net(board_tensor)
        feature_latent = self.feature_net(feature_tensor)
        return self.combined(th.cat([board_latent, feature_latent], dim=1))


def make_env(size):
    return SnakeEnv(size=size)


def evaluate_model(model, env, episodes):
    snake_lengths = []
    step_counts = []
    rewards = []
    events = {}

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        snake_lengths.append(info["snake_len"])
        step_counts.append(info["steps"])
        rewards.append(episode_reward)
        events[info["event"]] = events.get(info["event"], 0) + 1

    avg_snake_length = float(np.mean(snake_lengths))
    avg_steps = float(np.mean(step_counts))
    score = (avg_snake_length ** 2) / max(avg_steps, 1.0)
    selection_score = (avg_snake_length * 1000.0) - avg_steps

    return {
        "avg_snake_length": avg_snake_length,
        "avg_steps": avg_steps,
        "avg_reward": float(np.mean(rewards)),
        "max_snake_length": int(np.max(snake_lengths)),
        "median_snake_length": float(np.median(snake_lengths)),
        "score": score,
        "selection_score": selection_score,
        "terminal_events": events,
    }


class BenchmarkCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, eval_episodes, best_model_path, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.best_model_path = Path(best_model_path)
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_score = float("-inf")
        self.history = []

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True

        metrics = evaluate_model(self.model, self.eval_env, self.eval_episodes)
        metrics["timesteps"] = self.num_timesteps
        self.history.append(metrics)

        is_best = metrics["selection_score"] > self.best_score
        if is_best:
            self.best_score = metrics["selection_score"]
            self.model.save(self.best_model_path)

        if self.verbose:
            print(
                "eval",
                json.dumps(
                    {
                        "timesteps": metrics["timesteps"],
                        "avg_snake_length": round(metrics["avg_snake_length"], 3),
                        "avg_steps": round(metrics["avg_steps"], 3),
                        "selection_score": round(metrics["selection_score"], 3),
                        "score": round(metrics["score"], 6),
                        "best": is_best,
                    }
                ),
            )

        return True


def build_model(env, seed):
    policy_kwargs = {
        "features_extractor_class": EgocentricExtractor,
        "features_extractor_kwargs": {"features_dim": 192},
        "net_arch": [256, 256],
    }

    return DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2_000,
        exploration_fraction=0.40,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        max_grad_norm=10.0,
        stats_window_size=50,
        tensorboard_log=None,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device="auto",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--benchmark-episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/latest_run"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.size)
    eval_env = make_env(args.size)

    callback = BenchmarkCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        best_model_path=args.output_dir / "best_model",
        verbose=1,
    )

    model = build_model(env, seed=args.seed)
    print("train start...")
    model.learn(total_timesteps=args.train_steps, callback=callback, progress_bar=False)

    best_model_zip = args.output_dir / "best_model.zip"
    if not best_model_zip.exists():
        model.save(args.output_dir / "best_model")

    best_model = DQN.load(best_model_zip)
    benchmark_env = make_env(args.size)
    benchmark = evaluate_model(best_model, benchmark_env, args.benchmark_episodes)

    results = {
        "config": {
            "size": args.size,
            "train_steps": args.train_steps,
            "eval_freq": args.eval_freq,
            "eval_episodes": args.eval_episodes,
            "benchmark_episodes": args.benchmark_episodes,
            "seed": args.seed,
        },
        "best_eval_selection_score": callback.best_score,
        "eval_history": callback.history,
        "benchmark": benchmark,
    }

    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print()
    print("benchmark results:")
    print(json.dumps(benchmark, indent=2))

    benchmark_env.close()
    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
