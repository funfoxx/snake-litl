from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from snake.util.sumtree import SumTree
from snake_env import SnakeEnv

ROOT = Path(__file__).resolve().parent


@dataclass(slots=True)
class RunConfig:
    iteration: int = 3
    run_name: str = "main"
    total_timesteps: int = 200_000
    eval_freq: int = 5_000
    eval_episodes: int = 50
    benchmark_episodes: int = 1_000
    seed: int = 7
    learning_rate: float = 3e-4
    buffer_size: int = 150_000
    learning_starts: int = 5_000
    batch_size: int = 256
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 2_500
    exploration_fraction: float = 0.35
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.03
    hidden_sizes: tuple[int, ...] = (256, 256, 128)
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    priority_epsilon: float = 1e-3
    max_grad_norm: float = 10.0
    export_best: bool = False
    device: str = "cpu"


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=3)
    parser.add_argument("--run-name", type=str, default="main")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--benchmark-episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=150_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=2_500)
    parser.add_argument("--exploration-fraction", type=float, default=0.35)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.03)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-end", type=float, default=1.0)
    parser.add_argument("--priority-epsilon", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cpu")
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
        hidden_sizes=tuple(args.hidden_sizes),
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        priority_epsilon=args.priority_epsilon,
        max_grad_norm=args.max_grad_norm,
        export_best=args.export_best,
        device=args.device,
    )


def iteration_artifact_dir(config: RunConfig) -> Path:
    base_dir = ROOT / "artifacts" / f"iteration{config.iteration}"
    if config.run_name == "main":
        return base_dir
    return base_dir / config.run_name


def score_from_stats(avg_length: float, avg_steps: float) -> float:
    return 0.0 if avg_steps <= 0 else (avg_length**2) / avg_steps


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def linear_schedule(step: int, duration: int, start: float, end: float) -> float:
    if duration <= 0:
        return end
    progress = min(max(step / duration, 0.0), 1.0)
    return start + progress * (end - start)


class PrioritizedReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        capacity: int,
        alpha: float,
        priority_epsilon: float,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.tree = SumTree(capacity)
        self.write_index = 0
        self.size = 0
        self.max_priority = self._priority(1.0)

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def _priority(self, td_error: float) -> float:
        return float((abs(td_error) + self.priority_epsilon) ** self.alpha)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.write_index
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        tree_idx = idx + self.capacity - 1
        self.tree.data[idx] = idx
        self.tree.update(tree_idx, self.max_priority)

        self.write_index = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> dict[str, np.ndarray]:
        if self.size < batch_size:
            raise ValueError("not enough replay samples available")

        total_priority = self.tree.sum()
        if total_priority <= 0.0:
            raise ValueError("invalid replay priority sum")

        batch_indices = np.zeros((batch_size,), dtype=np.int64)
        tree_indices = np.zeros((batch_size,), dtype=np.int64)
        priorities = np.zeros((batch_size,), dtype=np.float32)

        segment = total_priority / batch_size
        for sample_idx in range(batch_size):
            start = segment * sample_idx
            end = segment * (sample_idx + 1)
            value = np.random.uniform(start, end)
            tree_idx, priority, data_idx = self.tree.retrieve(value)
            batch_indices[sample_idx] = data_idx
            tree_indices[sample_idx] = tree_idx
            priorities[sample_idx] = priority

        probs = priorities / total_priority
        weights = np.power(self.size * probs, -beta)
        weights /= np.max(weights)

        return {
            "states": self.states[batch_indices],
            "actions": self.actions[batch_indices],
            "rewards": self.rewards[batch_indices],
            "next_states": self.next_states[batch_indices],
            "dones": self.dones[batch_indices],
            "weights": weights.astype(np.float32),
            "tree_indices": tree_indices,
        }

    def update_priorities(
        self,
        tree_indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        for tree_idx, td_error in zip(tree_indices, td_errors):
            priority = self._priority(float(td_error))
            self.tree.update(int(tree_idx), priority)
            if priority > self.max_priority:
                self.max_priority = priority


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...],
    ):
        super().__init__()

        layers: list[nn.Module] = []
        last_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(nn.ReLU())
            last_dim = hidden_size
        self.backbone = nn.Sequential(*layers)

        head_dim = max(64, last_dim // 2)
        self.value_head = nn.Sequential(
            nn.Linear(last_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(last_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, action_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        values = self.value_head(features)
        advantages = self.advantage_head(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: RunConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = config.hidden_sizes
        self.gamma = config.gamma
        self.max_grad_norm = config.max_grad_norm
        self.device = torch.device(config.device)

        self.online_net = DuelingQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
        ).to(self.device)
        self.target_net = DuelingQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.learning_rate)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor.unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())

    def train_step(
        self,
        replay_buffer: PrioritizedReplayBuffer,
        batch_size: int,
        beta: float,
    ) -> dict[str, float]:
        batch = replay_buffer.sample(batch_size, beta)

        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            batch["next_states"], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)

        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        td_errors = targets - q_values
        losses = F.smooth_l1_loss(q_values, targets, reduction="none")
        loss = torch.mean(weights * losses)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        replay_buffer.update_priorities(
            batch["tree_indices"],
            td_errors.detach().abs().cpu().numpy(),
        )

        return {
            "loss": float(loss.item()),
            "mean_abs_td_error": float(td_errors.detach().abs().mean().item()),
            "mean_q": float(q_values.detach().mean().item()),
        }

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: Path, config: RunConfig, metadata: dict[str, float]) -> None:
        payload = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "model_state_dict": self.online_net.state_dict(),
            "config": asdict(config),
            "metadata": metadata,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "DQNAgent":
        checkpoint = torch.load(path, map_location=device)
        config = RunConfig(
            hidden_sizes=tuple(checkpoint["hidden_sizes"]),
            device=device,
        )
        agent = cls(
            state_dim=int(checkpoint["state_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            config=config,
        )
        agent.online_net.load_state_dict(checkpoint["model_state_dict"])
        agent.sync_target()
        return agent


def evaluate_agent(
    agent: DQNAgent,
    env: SnakeEnv,
    episodes: int,
) -> dict[str, float]:
    snake_lengths = []
    steps = []

    for episode_seed in range(episodes):
        obs, _ = env.reset(seed=episode_seed)
        done = False
        while not done:
            action = agent.select_action(obs, epsilon=0.0)
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


def train(config: RunConfig) -> dict[str, object]:
    artifact_dir = iteration_artifact_dir(config)
    best_model_path = artifact_dir / "best_model.zip"
    summary_path = artifact_dir / "summary.json"
    best_export_path = ROOT / "best" / f"iteration{config.iteration}.zip"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_model_path.unlink(missing_ok=True)

    set_global_seeds(config.seed)
    env = SnakeEnv()
    eval_env = SnakeEnv()
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, config=config)
    replay_buffer = PrioritizedReplayBuffer(
        state_dim=state_dim,
        capacity=config.buffer_size,
        alpha=config.per_alpha,
        priority_epsilon=config.priority_epsilon,
    )

    evaluations: list[dict[str, float]] = []
    training_stats: list[dict[str, float]] = []
    best_score = float("-inf")
    best_timestep = 0

    obs, _ = env.reset(seed=config.seed)
    exploration_steps = int(config.total_timesteps * config.exploration_fraction)

    print(
        f"training iteration {config.iteration} "
        f"run={config.run_name} "
        f"for {config.total_timesteps} timesteps"
    )

    for timestep in range(1, config.total_timesteps + 1):
        epsilon = linear_schedule(
            step=timestep - 1,
            duration=exploration_steps,
            start=config.exploration_initial_eps,
            end=config.exploration_final_eps,
        )
        beta = linear_schedule(
            step=timestep - 1,
            duration=config.total_timesteps,
            start=config.per_beta_start,
            end=config.per_beta_end,
        )

        action = agent.select_action(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs

        if done:
            obs, _ = env.reset()

        if (
            timestep >= config.learning_starts
            and timestep % config.train_freq == 0
            and len(replay_buffer) >= config.batch_size
        ):
            for _ in range(config.gradient_steps):
                train_metrics = agent.train_step(
                    replay_buffer=replay_buffer,
                    batch_size=config.batch_size,
                    beta=beta,
                )
                train_metrics["timesteps"] = float(timestep)
                training_stats.append(train_metrics)

        if timestep % config.target_update_interval == 0:
            agent.sync_target()

        if timestep % config.eval_freq == 0:
            metrics = evaluate_agent(agent, eval_env, config.eval_episodes)
            metrics["timesteps"] = int(timestep)
            metrics["epsilon"] = float(epsilon)
            metrics["beta"] = float(beta)
            if training_stats:
                recent = training_stats[-100:]
                metrics["mean_loss"] = float(np.mean([item["loss"] for item in recent]))
                metrics["mean_abs_td_error"] = float(
                    np.mean([item["mean_abs_td_error"] for item in recent])
                )
            evaluations.append(metrics)

            if metrics["score"] > best_score:
                best_score = metrics["score"]
                best_timestep = int(timestep)
                agent.save(
                    best_model_path,
                    config=config,
                    metadata={
                        "best_eval_score": best_score,
                        "best_eval_timestep": best_timestep,
                    },
                )

            print(
                "[eval] "
                f"t={metrics['timesteps']} "
                f"avg_len={metrics['avg_length']:.3f} "
                f"avg_steps={metrics['avg_steps']:.3f} "
                f"score={metrics['score']:.6f} "
                f"best={best_score:.6f}"
            )

    if not best_model_path.exists():
        best_timestep = config.total_timesteps
        last_metrics = evaluate_agent(agent, eval_env, config.eval_episodes)
        best_score = last_metrics["score"]
        agent.save(
            best_model_path,
            config=config,
            metadata={
                "best_eval_score": best_score,
                "best_eval_timestep": best_timestep,
            },
        )

    best_agent = DQNAgent.load(best_model_path, device=config.device)
    benchmark_env = SnakeEnv()
    benchmark = evaluate_agent(best_agent, benchmark_env, config.benchmark_episodes)

    summary = {
        "config": asdict(config),
        "best_eval_score": best_score,
        "best_eval_timestep": best_timestep,
        "evaluations": evaluations,
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
    print(f"best eval timestep: {best_timestep}")
    print(f"best eval score: {best_score:.9f}")
    if config.export_best:
        print(f"exported model: {best_export_path}")

    env.close()
    eval_env.close()
    benchmark_env.close()

    return summary


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
