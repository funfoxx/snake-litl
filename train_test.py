import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from snake_env import SnakeEnv


EVAL_DIR = ROOT / "eval" / "iteration1"
BEST_MODEL_PATH = EVAL_DIR / "best_model.zip"
BEST_ITERATION_PATH = ROOT / "best" / "iteration1.zip"


def make_env():
    return Monitor(SnakeEnv())


def train_model(total_timesteps, eval_freq, eval_episodes, seed):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(make_env, n_envs=8, seed=seed)
    eval_env = make_env()
    callback_eval_freq = max(eval_freq // env.num_envs, 1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(EVAL_DIR),
        log_path=str(EVAL_DIR),
        eval_freq=callback_eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "activation_fn": th.nn.ReLU,
            "net_arch": [128, 128],
        },
        verbose=1,
        seed=seed,
        device="auto",
    )

    print("train start...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=False)

    eval_env.close()
    env.close()


def evaluate_model(model_path, episodes, seed):
    model = PPO.load(model_path)
    env = SnakeEnv()

    snake_len = []
    steps = []
    scores = []

    print("test start...")
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

        snake_len.append(info["snake_len"])
        steps.append(info["steps"])
        scores.append(info["score"])

    env.close()
    return {
        "episodes": episodes,
        "avg_snake_length": float(np.mean(snake_len)),
        "avg_steps": float(np.mean(steps)),
        "score": float((np.mean(snake_len) ** 2) / np.mean(steps)),
        "avg_episode_score": float(np.mean(scores)),
        "max_snake_length": int(np.max(snake_len)),
        "min_steps": int(np.min(steps)),
        "max_steps": int(np.max(steps)),
    }


def read_best_mean_reward():
    evaluations_path = EVAL_DIR / "evaluations.npz"
    if not evaluations_path.exists():
        return None

    evaluations = np.load(evaluations_path)
    rewards = evaluations["results"]
    timesteps = evaluations["timesteps"]
    mean_rewards = rewards.mean(axis=1)
    best_index = int(np.argmax(mean_rewards))
    return {
        "best_mean_reward": float(mean_rewards[best_index]),
        "best_timestep": int(timesteps[best_index]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    train_model(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Best model not found at {BEST_MODEL_PATH}")

    metrics = evaluate_model(BEST_MODEL_PATH, args.test_episodes, args.seed)
    best_stats = read_best_mean_reward()
    if best_stats is not None:
        metrics.update(best_stats)

    shutil.copyfile(BEST_MODEL_PATH, BEST_ITERATION_PATH)

    print()
    print("test results:")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
