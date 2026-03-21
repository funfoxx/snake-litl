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
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from snake.solver.greedy import GreedySolver
from snake_env import LEFT_OF, RIGHT_OF, SnakeEnv


def make_env():
    return Monitor(SnakeEnv())


def get_artifact_paths(run_name):
    eval_dir = ROOT / "eval" / run_name
    return {
        "eval_dir": eval_dir,
        "best_model_path": eval_dir / "best_model.zip",
        "best_model_stem": eval_dir / "best_model",
        "best_iteration_path": ROOT / "best" / f"{run_name}.zip",
    }


def absolute_to_relative_action(current_direction, next_direction):
    if next_direction == current_direction:
        return 1
    if LEFT_OF[current_direction] == next_direction:
        return 0
    if RIGHT_OF[current_direction] == next_direction:
        return 2
    raise ValueError(
        f"Solver produced invalid reverse move from {current_direction} to {next_direction}"
    )


def run_policy_episodes(model, env, episodes, seed):
    snake_len = []
    steps = []
    scores = []
    rewards = []

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        snake_len.append(info["snake_len"])
        steps.append(info["steps"])
        scores.append(info["score"])
        rewards.append(episode_reward)

    avg_snake_length = float(np.mean(snake_len))
    avg_steps = float(np.mean(steps))
    return {
        "episodes": episodes,
        "avg_reward": float(np.mean(rewards)),
        "avg_snake_length": avg_snake_length,
        "avg_steps": avg_steps,
        "score": float((avg_snake_length**2) / avg_steps),
        "avg_episode_score": float(np.mean(scores)),
        "max_snake_length": int(np.max(snake_len)),
        "min_steps": int(np.min(steps)),
        "max_steps": int(np.max(steps)),
    }


class ScoreEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq,
        eval_episodes,
        eval_seed,
        best_model_path,
        log_path,
        verbose=1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.best_model_path = Path(best_model_path)
        self.log_path = Path(log_path)
        self.best_score = -np.inf
        self.history = {
            "timesteps": [],
            "avg_rewards": [],
            "avg_lengths": [],
            "avg_steps": [],
            "scores": [],
        }

    def _record_metrics(self, timestep, metrics):
        self.history["timesteps"].append(int(timestep))
        self.history["avg_rewards"].append(float(metrics["avg_reward"]))
        self.history["avg_lengths"].append(float(metrics["avg_snake_length"]))
        self.history["avg_steps"].append(float(metrics["avg_steps"]))
        self.history["scores"].append(float(metrics["score"]))
        np.savez(
            self.log_path,
            timesteps=np.array(self.history["timesteps"], dtype=np.int64),
            avg_rewards=np.array(self.history["avg_rewards"], dtype=np.float32),
            avg_lengths=np.array(self.history["avg_lengths"], dtype=np.float32),
            avg_steps=np.array(self.history["avg_steps"], dtype=np.float32),
            scores=np.array(self.history["scores"], dtype=np.float32),
        )

    def evaluate_and_save(self, model, timestep, label):
        metrics = run_policy_episodes(
            model=model,
            env=self.eval_env,
            episodes=self.eval_episodes,
            seed=self.eval_seed,
        )
        self._record_metrics(timestep, metrics)

        if self.verbose:
            print(
                f"[{label}] timestep={timestep} score={metrics['score']:.6f} "
                f"avg_len={metrics['avg_snake_length']:.3f} avg_steps={metrics['avg_steps']:.3f} "
                f"avg_reward={metrics['avg_reward']:.3f}"
            )

        if metrics["score"] > self.best_score:
            self.best_score = metrics["score"]
            model.save(str(self.best_model_path))
            if self.verbose:
                print(f"[{label}] new best score {self.best_score:.6f}")

        return metrics

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.evaluate_and_save(self.model, self.num_timesteps, label="eval")
        return True


def collect_expert_dataset(episodes, seed):
    env = SnakeEnv()
    observations = []
    actions = []
    lengths = []
    steps = []
    scores = []

    print("collecting greedy demonstrations...")
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        solver = GreedySolver(env.snake)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            direction = solver.next_direc()
            action = absolute_to_relative_action(env.snake.direc, direction)
            observations.append(obs.copy())
            actions.append(action)

            obs, _, terminated, truncated, info = env.step(action)
            solver.snake = env.snake

        lengths.append(info["snake_len"])
        steps.append(info["steps"])
        scores.append(info["score"])

    env.close()

    avg_length = float(np.mean(lengths))
    avg_steps = float(np.mean(steps))
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        {
            "expert_episodes": episodes,
            "expert_transitions": int(len(actions)),
            "expert_avg_snake_length": avg_length,
            "expert_avg_steps": avg_steps,
            "expert_score": float((avg_length**2) / avg_steps),
            "expert_avg_episode_score": float(np.mean(scores)),
            "expert_max_snake_length": int(np.max(lengths)),
        },
    )


def collect_dagger_dataset(model, episodes, seed):
    env = SnakeEnv()
    observations = []
    actions = []
    lengths = []
    steps = []
    scores = []

    print("collecting dagger relabels...")
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        solver = GreedySolver(env.snake)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            direction = solver.next_direc()
            expert_action = absolute_to_relative_action(env.snake.direc, direction)
            observations.append(obs.copy())
            actions.append(expert_action)

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            solver.snake = env.snake

        lengths.append(info["snake_len"])
        steps.append(info["steps"])
        scores.append(info["score"])

    env.close()

    avg_length = float(np.mean(lengths))
    avg_steps = float(np.mean(steps))
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        {
            "dagger_episodes": episodes,
            "dagger_transitions": int(len(actions)),
            "dagger_rollout_avg_snake_length": avg_length,
            "dagger_rollout_avg_steps": avg_steps,
            "dagger_rollout_score": float((avg_length**2) / avg_steps),
            "dagger_rollout_avg_episode_score": float(np.mean(scores)),
            "dagger_rollout_max_snake_length": int(np.max(lengths)),
        },
    )


def _batched_logits(model, observations, batch_size):
    logits = []
    with th.no_grad():
        for start in range(0, len(observations), batch_size):
            batch_obs = th.as_tensor(
                observations[start : start + batch_size],
                dtype=th.float32,
                device=model.device,
            )
            batch_logits = model.policy.get_distribution(batch_obs).distribution.logits
            logits.append(batch_logits)
    return th.cat(logits, dim=0)


def pretrain_policy(model, observations, actions, epochs, batch_size, seed):
    rng = np.random.default_rng(seed)
    num_samples = len(actions)
    indices = rng.permutation(num_samples)
    val_size = max(int(0.1 * num_samples), 1024)
    val_size = min(val_size, max(num_samples // 5, 1))
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    train_obs = observations[train_idx]
    train_actions = actions[train_idx]
    val_obs = observations[val_idx]
    val_actions = actions[val_idx]
    class_counts = np.bincount(train_actions, minlength=model.action_space.n)
    class_weights = class_counts.sum() / np.clip(class_counts, 1, None)
    class_weights = class_weights / class_weights.mean()
    class_weights_tensor = th.as_tensor(
        class_weights,
        dtype=th.float32,
        device=model.device,
    )

    history = []
    print("behavior cloning pretraining...")

    for epoch in range(1, epochs + 1):
        model.policy.set_training_mode(True)
        permutation = rng.permutation(len(train_actions))
        total_loss = 0.0

        for start in range(0, len(permutation), batch_size):
            batch_ids = permutation[start : start + batch_size]
            batch_obs = th.as_tensor(
                train_obs[batch_ids],
                dtype=th.float32,
                device=model.device,
            )
            batch_actions = th.as_tensor(
                train_actions[batch_ids],
                dtype=th.long,
                device=model.device,
            )

            logits = model.policy.get_distribution(batch_obs).distribution.logits
            loss = F.cross_entropy(logits, batch_actions, weight=class_weights_tensor)

            model.policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            model.policy.optimizer.step()

            total_loss += loss.item() * len(batch_ids)

        model.policy.set_training_mode(False)
        val_logits = _batched_logits(model, val_obs, batch_size)
        val_actions_tensor = th.as_tensor(val_actions, dtype=th.long, device=model.device)
        val_loss = F.cross_entropy(
            val_logits,
            val_actions_tensor,
            weight=class_weights_tensor,
        ).item()
        val_acc = float((val_logits.argmax(dim=1) == val_actions_tensor).float().mean().item())
        train_loss = total_loss / len(train_actions)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_accuracy": val_acc,
                "class_weights": class_weights.tolist(),
            }
        )
        print(
            f"bc epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    return history[-1]


def warmstart_policy_with_dagger(
    model,
    seed,
    bc_episodes,
    bc_epochs,
    bc_batch_size,
    dagger_rounds,
    dagger_episodes,
):
    observations, actions, expert_stats = collect_expert_dataset(bc_episodes, seed + 10_000)
    bc_stats = pretrain_policy(
        model=model,
        observations=observations,
        actions=actions,
        epochs=bc_epochs,
        batch_size=bc_batch_size,
        seed=seed,
    )

    dagger_history = []
    aggregated_observations = observations
    aggregated_actions = actions

    effective_dagger_rounds = dagger_rounds if dagger_episodes > 0 else 0
    for round_idx in range(1, effective_dagger_rounds + 1):
        round_observations, round_actions, round_stats = collect_dagger_dataset(
            model=model,
            episodes=dagger_episodes,
            seed=seed + 20_000 + (round_idx * 1_000),
        )
        aggregated_observations = np.concatenate((aggregated_observations, round_observations))
        aggregated_actions = np.concatenate((aggregated_actions, round_actions))
        bc_stats = pretrain_policy(
            model=model,
            observations=aggregated_observations,
            actions=aggregated_actions,
            epochs=bc_epochs,
            batch_size=bc_batch_size,
            seed=seed + round_idx,
        )
        dagger_history.append(
            {
                "round": round_idx,
                "aggregate_transitions": int(len(aggregated_actions)),
                **round_stats,
                **{f"bc_{key}": value for key, value in bc_stats.items()},
            }
        )

    return {
        **expert_stats,
        **{f"bc_{key}": value for key, value in bc_stats.items()},
        "dagger_rounds_completed": effective_dagger_rounds,
        "dagger_history": dagger_history,
        "warmstart_transitions": int(len(aggregated_actions)),
    }


def train_model(
    total_timesteps,
    eval_freq,
    eval_episodes,
    seed,
    run_name,
    bc_episodes,
    bc_epochs,
    bc_batch_size,
    dagger_rounds,
    dagger_episodes,
    n_envs,
    learning_rate,
    n_steps,
    batch_size,
    n_epochs,
    ent_coef,
):
    artifact_paths = get_artifact_paths(run_name)
    artifact_paths["eval_dir"].mkdir(parents=True, exist_ok=True)
    do_bc = bc_episodes > 0 and bc_epochs > 0
    warmstart_stats = {}

    env = make_vec_env(make_env, n_envs=n_envs, seed=seed)
    eval_env = SnakeEnv()
    callback_eval_freq = max(eval_freq // env.num_envs, 1)

    score_callback = ScoreEvalCallback(
        eval_env=eval_env,
        eval_freq=callback_eval_freq,
        eval_episodes=eval_episodes,
        eval_seed=seed + 100_000,
        best_model_path=artifact_paths["best_model_stem"],
        log_path=artifact_paths["eval_dir"] / "evaluations.npz",
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "activation_fn": th.nn.ReLU,
            "net_arch": [256, 256, 256],
        },
        verbose=1,
        seed=seed,
        device="auto",
    )

    if do_bc:
        warmstart_stats = warmstart_policy_with_dagger(
            model=model,
            seed=seed,
            bc_episodes=bc_episodes,
            bc_epochs=bc_epochs,
            bc_batch_size=bc_batch_size,
            dagger_rounds=dagger_rounds,
            dagger_episodes=dagger_episodes,
        )
        score_callback.evaluate_and_save(model, timestep=0, label="bc")
    else:
        score_callback.evaluate_and_save(model, timestep=0, label="init")

    print("ppo fine-tuning...")
    model.learn(total_timesteps=total_timesteps, callback=score_callback, progress_bar=False)

    eval_env.close()
    env.close()
    return {
        "run_name": run_name,
        **warmstart_stats,
    }


def evaluate_model(model_path, episodes, seed):
    model = PPO.load(model_path)
    env = SnakeEnv()
    metrics = run_policy_episodes(model=model, env=env, episodes=episodes, seed=seed)
    env.close()
    return metrics


def read_best_eval_stats(run_name):
    evaluations_path = get_artifact_paths(run_name)["eval_dir"] / "evaluations.npz"
    if not evaluations_path.exists():
        return None

    evaluations = np.load(evaluations_path)
    scores = evaluations["scores"]
    timesteps = evaluations["timesteps"]
    avg_rewards = evaluations["avg_rewards"]
    avg_lengths = evaluations["avg_lengths"]
    avg_steps = evaluations["avg_steps"]
    best_index = int(np.argmax(scores))
    return {
        "best_score_eval": float(scores[best_index]),
        "best_timestep": int(timesteps[best_index]),
        "best_eval_avg_reward": float(avg_rewards[best_index]),
        "best_eval_avg_snake_length": float(avg_lengths[best_index]),
        "best_eval_avg_steps": float(avg_steps[best_index]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="iteration3")
    parser.add_argument("--timesteps", type=int, default=5_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--test-episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--bc-episodes", type=int, default=200)
    parser.add_argument("--bc-epochs", type=int, default=5)
    parser.add_argument("--bc-batch-size", type=int, default=2048)
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--dagger-episodes", type=int, default=30)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--ent-coef", type=float, default=5e-4)
    args = parser.parse_args()

    train_stats = train_model(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        run_name=args.run_name,
        bc_episodes=args.bc_episodes,
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        dagger_rounds=args.dagger_rounds,
        dagger_episodes=args.dagger_episodes,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
    )

    artifact_paths = get_artifact_paths(args.run_name)
    if not artifact_paths["best_model_path"].exists():
        raise FileNotFoundError(f"Best model not found at {artifact_paths['best_model_path']}")

    metrics = evaluate_model(artifact_paths["best_model_path"], args.test_episodes, args.seed)
    best_stats = read_best_eval_stats(args.run_name)
    if best_stats is not None:
        metrics.update(best_stats)
    metrics.update(train_stats)

    shutil.copyfile(artifact_paths["best_model_path"], artifact_paths["best_iteration_path"])

    print()
    print("test results:")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
