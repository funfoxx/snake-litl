from __future__ import annotations

from collections import defaultdict

import gymnasium as gym
import numpy as np

from snake.base import Direc, Map, PointType, Pos, Snake

TURN_LEFT = {
    Direc.LEFT: Direc.DOWN,
    Direc.UP: Direc.LEFT,
    Direc.RIGHT: Direc.UP,
    Direc.DOWN: Direc.RIGHT,
}

TURN_RIGHT = {
    Direc.LEFT: Direc.UP,
    Direc.UP: Direc.RIGHT,
    Direc.RIGHT: Direc.DOWN,
    Direc.DOWN: Direc.LEFT,
}


class SnakeEnv(gym.Env):
    def __init__(self, size: int = 10):
        self.size = size
        self.capacity = (self.size - 2) * (self.size - 2)

        # 5x5 egocentric local view + relative food vector + length ratio +
        # starvation budget ratio + safe left/straight/right flags.
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(32,),
            dtype=np.float32,
        )
        # Relative actions: turn left, go straight, turn right.
        self.action_space = gym.spaces.Discrete(3)

    def _get_info(self):
        return {
            "snake_len": self.snake.len(),
            "steps": self.steps,
            "steps_since_food": self.steps_since_food,
        }

    def _action_to_direction(self, action: int) -> Direc:
        if action == 0:
            return TURN_LEFT[self.snake.direc]
        if action == 1:
            return self.snake.direc
        if action == 2:
            return TURN_RIGHT[self.snake.direc]
        raise ValueError(f"unknown action {action}")

    def _relative_to_absolute(self, row_offset: int, col_offset: int) -> Pos:
        head = self.snake.head()
        if self.snake.direc == Direc.UP:
            dx, dy = row_offset, col_offset
        elif self.snake.direc == Direc.RIGHT:
            dx, dy = col_offset, -row_offset
        elif self.snake.direc == Direc.DOWN:
            dx, dy = -row_offset, -col_offset
        else:
            dx, dy = -col_offset, row_offset
        return Pos(head.x + dx, head.y + dy)

    def _food_vector(self) -> tuple[float, float]:
        head = self.snake.head()
        food = self.snake_map.food
        if food is None:
            return 0.0, 0.0

        dx = food.x - head.x
        dy = food.y - head.y

        if self.snake.direc == Direc.UP:
            row_delta, col_delta = dx, dy
        elif self.snake.direc == Direc.RIGHT:
            row_delta, col_delta = -dy, dx
        elif self.snake.direc == Direc.DOWN:
            row_delta, col_delta = -dx, -dy
        else:
            row_delta, col_delta = dy, -dx

        scale = float(self.size - 2)
        forward = float(np.clip(-row_delta / scale, -1.0, 1.0))
        lateral = float(np.clip(col_delta / scale, -1.0, 1.0))
        return forward, lateral

    def _cell_value(self, pos: Pos) -> float:
        if (
            pos.x < 0
            or pos.x >= self.snake_map.num_rows
            or pos.y < 0
            or pos.y >= self.snake_map.num_cols
        ):
            return -1.0

        if pos == self.snake.head():
            return 0.5

        point_type = self.snake_map.point(pos).type
        if point_type == PointType.EMPTY:
            return 0.0
        if point_type == PointType.FOOD:
            return 1.0
        return -1.0

    def _safe_flag(self, direction: Direc) -> float:
        next_pos = self.snake.head().adj(direction)
        return 1.0 if self.snake_map.is_safe(next_pos) else -1.0

    def _max_steps_without_food(self) -> int:
        return max(self.capacity, self.snake.len() * 12)

    def _get_obs(self):
        local_view = []
        for row_offset in range(-2, 3):
            for col_offset in range(-2, 3):
                pos = self._relative_to_absolute(row_offset, col_offset)
                local_view.append(self._cell_value(pos))

        forward_food, lateral_food = self._food_vector()
        length_ratio = (self.snake.len() - 2) / max(1, self.capacity - 2)
        starvation_ratio = 1.0 - (
            self.steps_since_food / float(self._max_steps_without_food())
        )
        safe_left = self._safe_flag(TURN_LEFT[self.snake.direc])
        safe_straight = self._safe_flag(self.snake.direc)
        safe_right = self._safe_flag(TURN_RIGHT[self.snake.direc])

        return np.asarray(
            local_view
            + [
                forward_food,
                lateral_food,
                float(np.clip(length_ratio, 0.0, 1.0)),
                float(np.clip(starvation_ratio, -1.0, 1.0)),
                safe_left,
                safe_straight,
                safe_right,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.snake_map = Map(self.size, self.size)
        self.snake = Snake(self.snake_map)
        self.snake_map.create_rand_food()
        self.steps = 0
        self.steps_since_food = 0
        self.visit_counts = defaultdict(int)

        head = self.snake.head()
        self.visit_counts[(head.x, head.y)] = 1

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        direction = self._action_to_direction(int(action))
        head_before = self.snake.head()
        food_before = self.snake_map.food
        distance_before = Pos.manhattan_dist(head_before, food_before)
        length_before = self.snake.len()

        self.snake.move(direction)
        self.steps += 1

        got_food = self.snake.len() > length_before
        if got_food:
            self.steps_since_food = 0
        else:
            self.steps_since_food += 1

        if not self.snake_map.has_food():
            self.snake_map.create_rand_food()

        terminated = self.snake.dead or self.snake_map.is_full()
        truncated = self.steps_since_food >= self._max_steps_without_food()

        reward = -0.01
        if terminated:
            reward = 10.0 if self.snake_map.is_full() else -2.5
        elif truncated:
            reward = -1.0
        elif got_food:
            reward = 2.5 + 0.1 * self.snake.len()
        else:
            distance_after = Pos.manhattan_dist(self.snake.head(), self.snake_map.food)
            if distance_after < distance_before:
                reward += 0.15
            elif distance_after > distance_before:
                reward -= 0.15

            head = self.snake.head()
            visit_key = (head.x, head.y)
            self.visit_counts[visit_key] += 1
            reward -= 0.02 * min(self.visit_counts[visit_key] - 1, 4)

        observation = self._get_obs()
        info = self._get_info()
        info["got_food"] = got_food
        info["terminated_reason"] = "starved" if truncated and not terminated else None

        return observation, reward, terminated, truncated, info
