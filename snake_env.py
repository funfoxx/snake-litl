from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from snake.base import Direc, Map, PointType, Pos, Snake

RELATIVE_ACTIONS = {
    0: "left",
    1: "straight",
    2: "right",
}

TURN_LEFT = {
    Direc.UP: Direc.LEFT,
    Direc.LEFT: Direc.DOWN,
    Direc.DOWN: Direc.RIGHT,
    Direc.RIGHT: Direc.UP,
}

TURN_RIGHT = {
    Direc.UP: Direc.RIGHT,
    Direc.RIGHT: Direc.DOWN,
    Direc.DOWN: Direc.LEFT,
    Direc.LEFT: Direc.UP,
}

DIRECTION_VECTORS = {
    Direc.UP: (-1, 0),
    Direc.RIGHT: (0, 1),
    Direc.DOWN: (1, 0),
    Direc.LEFT: (0, -1),
}


@dataclass(frozen=True)
class SnakeEnvConfig:
    step_penalty: float = -0.02
    food_base_reward: float = 10.0
    food_length_scale: float = 0.2
    board_clear_bonus: float = 25.0
    death_penalty: float = -10.0
    closer_reward: float = 0.3
    farther_penalty: float = -0.3
    space_delta_scale: float = 0.2
    revisit_penalty: float = -0.2
    starvation_penalty: float = -3.0
    starvation_base: int = 40
    starvation_scale: int = 8
    no_progress_threshold: int = 0
    no_progress_penalty: float = -0.0


class SnakeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        size: int = 10,
        revisit_window: int = 12,
        config: SnakeEnvConfig | None = None,
    ):
        self.size = size
        self.inner_size = size - 2
        self.revisit_window = revisit_window
        self.config = config or SnakeEnvConfig()

        # 3 danger bits + 2 food coords + 1 length + 24 ray values
        # + 1 reachable-space ratio + 1 food-reachable flag + 2 tail coords
        # + 3 candidate actions * (blocked, food distance, reachable-space, food-reachable)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(46,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(len(RELATIVE_ACTIONS))

        self.snake_map: Map | None = None
        self.snake: Snake | None = None
        self.steps = 0
        self.steps_since_food = 0
        self.recent_heads: deque[tuple[int, int]] = deque(maxlen=self.revisit_window)
        self.last_food_distance = 0
        self.last_space_ratio = 0.0
        self.no_progress_steps = 0

    def _get_obs(self) -> np.ndarray:
        head = self.snake.head()
        if not self.snake_map.is_inside(head):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        food = self.snake.map.food

        forward = self.snake.direc
        left = TURN_LEFT[forward]
        right = TURN_RIGHT[forward]

        danger = np.array(
            [
                float(self._is_blocked(head.adj(forward))),
                float(self._is_blocked(head.adj(left))),
                float(self._is_blocked(head.adj(right))),
            ],
            dtype=np.float32,
        )

        food_dx = food.x - head.x
        food_dy = food.y - head.y
        food_forward, food_right = self._to_relative(food_dx, food_dy, forward)
        denom = float(self.inner_size)
        food_features = np.array(
            [food_forward / denom, food_right / denom],
            dtype=np.float32,
        )

        ray_features = []
        ray_dirs = self._relative_ray_directions(forward)
        max_ray = float(self.inner_size)
        for ray_dir in ray_dirs:
            wall_dist, body_dist, food_dist = self._scan_ray(head, ray_dir)
            ray_features.extend(
                [
                    wall_dist / max_ray,
                    body_dist / max_ray,
                    food_dist / max_ray,
                ]
            )

        length_feature = np.array(
            [self.snake.len() / float(self.snake.map.capacity)],
            dtype=np.float32,
        )

        current_space_ratio, current_food_reachable = self._reachable_region_features(head)
        structure_features = np.array(
            [current_space_ratio, current_food_reachable],
            dtype=np.float32,
        )

        tail = self.snake.tail()
        tail_dx = tail.x - head.x
        tail_dy = tail.y - head.y
        tail_forward, tail_right = self._to_relative(tail_dx, tail_dy, forward)
        tail_features = np.array(
            [tail_forward / denom, tail_right / denom],
            dtype=np.float32,
        )

        candidate_features = []
        max_food_dist = float(max(1, 2 * self.inner_size))
        for action in RELATIVE_ACTIONS:
            next_head = head.adj(self._resolve_action(action))
            blocked = float(self._is_blocked(next_head))
            if blocked:
                candidate_features.extend([1.0, 1.0, 0.0, 0.0])
                continue

            next_food_distance = Pos.manhattan_dist(next_head, food) / max_food_dist
            next_space_ratio, next_food_reachable = self._reachable_region_features(next_head)
            candidate_features.extend(
                [
                    blocked,
                    next_food_distance,
                    next_space_ratio,
                    next_food_reachable,
                ]
            )

        return np.concatenate(
            [
                danger,
                food_features,
                length_feature,
                np.asarray(ray_features, dtype=np.float32),
                structure_features,
                tail_features,
                np.asarray(candidate_features, dtype=np.float32),
            ]
        ).astype(np.float32)

    def _get_info(self) -> dict[str, int | float]:
        return {
            "snake_len": self.snake.len(),
            "steps": self.steps,
            "steps_since_food": self.steps_since_food,
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.snake_map = Map(self.size, self.size)
        self._seed_random_module()
        self.snake = Snake(self.snake_map)
        self._seed_random_module()
        self.snake_map.create_rand_food()

        self.steps = 0
        self.steps_since_food = 0
        self.recent_heads.clear()
        self.recent_heads.append((self.snake.head().x, self.snake.head().y))
        self.last_food_distance = self._food_distance()
        self.last_space_ratio, _ = self._reachable_region_features(self.snake.head())
        self.no_progress_steps = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        previous_distance = self.last_food_distance
        previous_space_ratio = self.last_space_ratio
        direction = self._resolve_action(int(action))
        len_before = self.snake.len()

        self.snake.move(direction)
        self.steps += 1
        self.steps_since_food += 1

        terminated = self.snake.dead
        truncated = False
        reward = self.config.step_penalty
        current_space_ratio = 0.0

        if not terminated:
            if not self.snake_map.has_food():
                self._seed_random_module()
                self.snake_map.create_rand_food()

            if self.snake_map.is_full():
                terminated = True
                reward += self.config.board_clear_bonus
            else:
                current_space_ratio, _ = self._reachable_region_features(self.snake.head())

        got_food = self.snake.len() > len_before
        if got_food:
            reward += self.config.food_base_reward + self.config.food_length_scale * self.snake.len()
            self.steps_since_food = 0
            self.no_progress_steps = 0
        elif terminated:
            reward += self.config.death_penalty
            self.no_progress_steps = 0
        else:
            current_distance = self._food_distance()
            if current_distance < previous_distance:
                reward += self.config.closer_reward
                self.no_progress_steps = 0
            elif current_distance > previous_distance:
                reward += self.config.farther_penalty
                self.no_progress_steps += 1
            else:
                self.no_progress_steps += 1

            reward += self.config.space_delta_scale * (current_space_ratio - previous_space_ratio)

            head_key = (self.snake.head().x, self.snake.head().y)
            if head_key in self.recent_heads:
                reward += self.config.revisit_penalty
            self.recent_heads.append(head_key)

            if (
                self.config.no_progress_threshold > 0
                and self.no_progress_steps >= self.config.no_progress_threshold
            ):
                reward += self.config.no_progress_penalty

            if self.steps_since_food >= self._max_steps_without_food():
                truncated = True
                reward += self.config.starvation_penalty

        self.last_food_distance = self._food_distance() if self.snake_map.is_inside(self.snake.head()) else 0
        self.last_space_ratio = current_space_ratio
        observation = self._get_obs()

        return observation, reward, terminated, truncated, self._get_info()

    def _resolve_action(self, action: int) -> Direc:
        current = self.snake.direc
        if RELATIVE_ACTIONS[action] == "left":
            return TURN_LEFT[current]
        if RELATIVE_ACTIONS[action] == "right":
            return TURN_RIGHT[current]
        return current

    def _is_blocked(self, pos: Pos) -> bool:
        return not self.snake_map.is_safe(pos)

    def _food_distance(self) -> int:
        return Pos.manhattan_dist(self.snake.head(), self.snake.map.food)

    def _seed_random_module(self) -> None:
        random.seed(int(self.np_random.integers(0, 2**32 - 1)))

    def _max_steps_without_food(self) -> int:
        return self.config.starvation_base + self.config.starvation_scale * self.snake.len()

    def _reachable_region_features(self, start: Pos) -> tuple[float, float]:
        if not self.snake_map.is_inside(start):
            return 0.0, 0.0

        if self._is_blocked(start) and start != self.snake.head():
            return 0.0, 0.0

        food = self.snake.map.food
        queue = deque([(start.x, start.y)])
        visited = {(start.x, start.y)}
        reachable = 0
        food_reachable = 0.0

        while queue:
            x, y = queue.popleft()
            reachable += 1
            if food is not None and x == food.x and y == food.y:
                food_reachable = 1.0

            for dx, dy in DIRECTION_VECTORS.values():
                nx = x + dx
                ny = y + dy
                if (nx, ny) in visited:
                    continue

                pos = Pos(nx, ny)
                if not self.snake_map.is_inside(pos):
                    continue

                point_type = self.snake_map.point(pos).type
                if point_type not in (PointType.EMPTY, PointType.FOOD):
                    continue

                visited.add((nx, ny))
                queue.append((nx, ny))

        return reachable / float(self.snake.map.capacity), food_reachable

    def _relative_ray_directions(self, forward: Direc) -> list[tuple[int, int]]:
        forward_delta, right_delta = self._forward_right_basis(forward)
        return [
            forward_delta,
            (forward_delta[0] + right_delta[0], forward_delta[1] + right_delta[1]),
            right_delta,
            (-forward_delta[0] + right_delta[0], -forward_delta[1] + right_delta[1]),
            (-forward_delta[0], -forward_delta[1]),
            (-forward_delta[0] - right_delta[0], -forward_delta[1] - right_delta[1]),
            (-right_delta[0], -right_delta[1]),
            (forward_delta[0] - right_delta[0], forward_delta[1] - right_delta[1]),
        ]

    def _scan_ray(self, start: Pos, delta: tuple[int, int]) -> tuple[float, float, float]:
        wall_dist = body_dist = food_dist = float(self.inner_size)

        x = start.x
        y = start.y
        step = 0
        while True:
            dx, dy = delta
            x += dx
            y += dy
            step += 1

            pos = Pos(x, y)
            point_type = self.snake_map.point(pos).type
            if point_type == PointType.WALL:
                wall_dist = float(step)
                break
            if point_type == PointType.FOOD and food_dist == float(self.inner_size):
                food_dist = float(step)
            elif point_type != PointType.EMPTY and point_type.value >= PointType.HEAD_L.value:
                if body_dist == float(self.inner_size):
                    body_dist = float(step)

        return wall_dist, body_dist, food_dist

    def _forward_right_basis(self, forward: Direc) -> tuple[tuple[int, int], tuple[int, int]]:
        forward_delta = DIRECTION_VECTORS[forward]
        right_delta = DIRECTION_VECTORS[TURN_RIGHT[forward]]
        return forward_delta, right_delta

    def _to_relative(self, dx: int, dy: int, forward: Direc) -> tuple[int, int]:
        if forward == Direc.UP:
            return -dx, dy
        if forward == Direc.RIGHT:
            return dy, dx
        if forward == Direc.DOWN:
            return dx, -dy
        return -dy, -dx
