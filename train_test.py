import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F

os.environ.setdefault("MPLCONFIGDIR", str((Path("artifacts") / "mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((Path("artifacts") / "cache").resolve()))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from snake.solver.greedy import GreedySolver
from snake_env import SnakeEnv, TURN_LEFT, TURN_RIGHT


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


def direction_to_relative_action(heading, next_direction):
    if next_direction == heading:
        return 1
    if TURN_LEFT[heading] == next_direction:
        return 0
    if TURN_RIGHT[heading] == next_direction:
        return 2
    raise ValueError(f"Teacher proposed invalid turn: {heading} -> {next_direction}")


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
    score = (avg_snake_length**2) / max(avg_steps, 1.0)
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


def collect_expert_dataset(size, episodes, seed):
    env = make_env(size)

    boards = []
    features = []
    actions = []
    episode_lengths = []
    episode_steps = []
    terminal_events = Counter()
    action_counts = Counter()

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            solver = GreedySolver(env.snake)
            done = False
            info = {}

            while not done:
                action = direction_to_relative_action(
                    env.snake.direc,
                    solver.next_direc(),
                )
                boards.append(obs["board"].copy())
                features.append(obs["features"].copy())
                actions.append(action)
                action_counts[action] += 1

                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            episode_lengths.append(info["snake_len"])
            episode_steps.append(info["steps"])
            terminal_events[info["event"]] += 1
    finally:
        env.close()

    return {
        "board": np.stack(boards).astype(np.float32),
        "features": np.stack(features).astype(np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "summary": {
            "episodes": episodes,
            "transitions": int(len(actions)),
            "avg_teacher_length": float(np.mean(episode_lengths)),
            "avg_teacher_steps": float(np.mean(episode_steps)),
            "max_teacher_length": int(np.max(episode_lengths)),
            "terminal_events": dict(terminal_events),
            "action_counts": {str(k): int(v) for k, v in sorted(action_counts.items())},
        },
    }


def collect_dagger_dataset(size, episodes, seed, model, teacher_beta):
    env = make_env(size)
    rng = np.random.default_rng(seed)

    boards = []
    features = []
    actions = []
    episode_lengths = []
    episode_steps = []
    terminal_events = Counter()
    label_action_counts = Counter()
    control_action_counts = Counter()
    control_source_counts = Counter()

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            solver = GreedySolver(env.snake)
            done = False
            info = {}

            while not done:
                teacher_action = direction_to_relative_action(
                    env.snake.direc,
                    solver.next_direc(),
                )
                boards.append(obs["board"].copy())
                features.append(obs["features"].copy())
                actions.append(teacher_action)
                label_action_counts[teacher_action] += 1

                use_teacher = rng.random() < teacher_beta
                if use_teacher:
                    action = teacher_action
                    control_source_counts["teacher"] += 1
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
                    control_source_counts["student"] += 1

                control_action_counts[int(action)] += 1
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            episode_lengths.append(info["snake_len"])
            episode_steps.append(info["steps"])
            terminal_events[info["event"]] += 1
    finally:
        env.close()

    total_control_steps = control_source_counts["teacher"] + control_source_counts["student"]

    return {
        "board": np.stack(boards).astype(np.float32),
        "features": np.stack(features).astype(np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "summary": {
            "episodes": episodes,
            "transitions": int(len(actions)),
            "teacher_beta": float(teacher_beta),
            "avg_rollout_length": float(np.mean(episode_lengths)),
            "avg_rollout_steps": float(np.mean(episode_steps)),
            "max_rollout_length": int(np.max(episode_lengths)),
            "teacher_control_rate": float(
                control_source_counts["teacher"] / max(total_control_steps, 1)
            ),
            "terminal_events": dict(terminal_events),
            "label_action_counts": {
                str(k): int(v) for k, v in sorted(label_action_counts.items())
            },
            "control_action_counts": {
                str(k): int(v) for k, v in sorted(control_action_counts.items())
            },
            "control_source_counts": dict(control_source_counts),
        },
    }


def merge_datasets(left, right):
    return {
        "board": np.concatenate([left["board"], right["board"]], axis=0),
        "features": np.concatenate([left["features"], right["features"]], axis=0),
        "actions": np.concatenate([left["actions"], right["actions"]], axis=0),
    }


def pretrain_from_expert(
    model,
    dataset,
    epochs,
    batch_size,
    learning_rate,
):
    optimizer = th.optim.Adam(model.policy.q_net.parameters(), lr=learning_rate)
    device = model.device
    num_samples = dataset["actions"].shape[0]
    action_counts = np.bincount(dataset["actions"], minlength=model.action_space.n)
    class_weights = action_counts.sum() / np.maximum(action_counts, 1)
    class_weights = class_weights / class_weights.mean()
    class_weights_tensor = th.as_tensor(class_weights, dtype=th.float32, device=device)

    epoch_history = []
    indices = np.arange(num_samples)
    model.policy.q_net.train()

    for epoch in range(epochs):
        np.random.shuffle(indices)
        losses = []
        accuracies = []

        for start in range(0, num_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            obs = {
                "board": th.as_tensor(dataset["board"][batch_idx], device=device),
                "features": th.as_tensor(dataset["features"][batch_idx], device=device),
            }
            labels = th.as_tensor(dataset["actions"][batch_idx], device=device)

            logits = model.policy.q_net(obs)
            loss = F.cross_entropy(logits, labels, weight=class_weights_tensor)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.q_net.parameters(), max_norm=10.0)
            optimizer.step()

            losses.append(float(loss.item()))
            accuracies.append(float((logits.argmax(dim=1) == labels).float().mean().item()))

        epoch_stats = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "accuracy": float(np.mean(accuracies)),
        }
        epoch_history.append(epoch_stats)
        print("imitation", json.dumps(epoch_stats))

    model.policy.q_net_target.load_state_dict(model.policy.q_net.state_dict())
    model.policy.set_training_mode(False)

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "class_weights": [float(weight) for weight in class_weights.tolist()],
        "history": epoch_history,
        "final_loss": epoch_history[-1]["loss"],
        "final_accuracy": epoch_history[-1]["accuracy"],
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

    def evaluate_now(self, label, timesteps):
        metrics = evaluate_model(self.model, self.eval_env, self.eval_episodes)
        metrics["timesteps"] = int(timesteps)
        metrics["label"] = label
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
                        "label": label,
                        "timesteps": metrics["timesteps"],
                        "avg_snake_length": round(metrics["avg_snake_length"], 3),
                        "avg_steps": round(metrics["avg_steps"], 3),
                        "selection_score": round(metrics["selection_score"], 3),
                        "score": round(metrics["score"], 6),
                        "best": is_best,
                    }
                ),
            )

        return metrics

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True

        self.evaluate_now(label="rl_eval", timesteps=self.num_timesteps)
        return True


def build_model(env, seed, warm_start=False):
    policy_kwargs = {
        "features_extractor_class": EgocentricExtractor,
        "features_extractor_kwargs": {"features_dim": 192},
        "net_arch": [256, 256],
    }

    learning_rate = 3e-4
    learning_starts = 1_000
    exploration_fraction = 0.40
    exploration_initial_eps = 1.0
    exploration_final_eps = 0.02

    if warm_start:
        learning_rate = 1e-4
        learning_starts = 0
        exploration_fraction = 0.10
        exploration_initial_eps = 0.05
        exploration_final_eps = 0.005

    return DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=100_000,
        learning_starts=learning_starts,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2_000,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
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
    parser.add_argument("--train-steps", type=int, default=30_000)
    parser.add_argument("--eval-freq", type=int, default=2_000)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--benchmark-episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/latest_run"))
    parser.add_argument("--imitation-episodes", type=int, default=120)
    parser.add_argument("--imitation-epochs", type=int, default=8)
    parser.add_argument("--imitation-batch-size", type=int, default=256)
    parser.add_argument("--imitation-lr", type=float, default=1e-3)
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--dagger-episodes", type=int, default=30)
    parser.add_argument("--dagger-beta-start", type=float, default=0.7)
    parser.add_argument("--dagger-beta-end", type=float, default=0.2)
    parser.add_argument("--dagger-epochs", type=int, default=3)
    parser.add_argument("--dagger-lr", type=float, default=5e-4)
    parser.add_argument("--skip-imitation", action="store_true")
    return parser.parse_args()


def interpolation_value(start, end, index, total):
    if total <= 1:
        return float(start)
    fraction = index / float(total - 1)
    return float(start + ((end - start) * fraction))


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

    model = build_model(env, seed=args.seed, warm_start=not args.skip_imitation)
    callback.init_callback(model)

    imitation_results = None
    expert_summary = None
    aggregation_results = []
    if not args.skip_imitation:
        print("collecting expert demonstrations...")
        expert_dataset = collect_expert_dataset(
            size=args.size,
            episodes=args.imitation_episodes,
            seed=args.seed,
        )
        expert_summary = expert_dataset["summary"]
        print("expert", json.dumps(expert_summary))

        print("pretraining from demonstrations...")
        imitation_results = pretrain_from_expert(
            model=model,
            dataset=expert_dataset,
            epochs=args.imitation_epochs,
            batch_size=args.imitation_batch_size,
            learning_rate=args.imitation_lr,
        )
        callback.evaluate_now(label="post_imitation", timesteps=0)

        aggregated_dataset = expert_dataset
        for round_idx in range(args.dagger_rounds):
            teacher_beta = interpolation_value(
                args.dagger_beta_start,
                args.dagger_beta_end,
                round_idx,
                args.dagger_rounds,
            )
            round_number = round_idx + 1

            print(f"collecting dagger round {round_number}...")
            dagger_dataset = collect_dagger_dataset(
                size=args.size,
                episodes=args.dagger_episodes,
                seed=args.seed + (round_number * 10_000),
                model=model,
                teacher_beta=teacher_beta,
            )
            print("dagger", json.dumps({"round": round_number, **dagger_dataset["summary"]}))

            aggregated_dataset = merge_datasets(aggregated_dataset, dagger_dataset)
            dagger_fit = pretrain_from_expert(
                model=model,
                dataset=aggregated_dataset,
                epochs=args.dagger_epochs,
                batch_size=args.imitation_batch_size,
                learning_rate=args.dagger_lr,
            )
            eval_metrics = callback.evaluate_now(
                label=f"dagger_round_{round_number}",
                timesteps=0,
            )
            aggregation_results.append(
                {
                    "round": round_number,
                    "combined_transitions": int(aggregated_dataset["actions"].shape[0]),
                    "rollout": dagger_dataset["summary"],
                    "train": dagger_fit,
                    "eval": eval_metrics,
                }
            )

    if args.train_steps > 0:
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
            "skip_imitation": args.skip_imitation,
            "imitation_episodes": args.imitation_episodes,
            "imitation_epochs": args.imitation_epochs,
            "imitation_batch_size": args.imitation_batch_size,
            "imitation_lr": args.imitation_lr,
            "dagger_rounds": args.dagger_rounds,
            "dagger_episodes": args.dagger_episodes,
            "dagger_beta_start": args.dagger_beta_start,
            "dagger_beta_end": args.dagger_beta_end,
            "dagger_epochs": args.dagger_epochs,
            "dagger_lr": args.dagger_lr,
        },
        "expert_summary": expert_summary,
        "imitation": imitation_results,
        "aggregation_rounds": aggregation_results,
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
