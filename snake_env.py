import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from snake.base import Direc, Map, PointType, Pos, Snake

RELATIVE_ACTIONS = {
    0: "left",
    1: "forward",
    2: "right",
}

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

ROTATIONS = {
    Direc.UP: 0,
    Direc.RIGHT: 1,
    Direc.DOWN: 2,
    Direc.LEFT: 3,
}


class SnakeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        size=10,
        food_reward=1.5,
        death_penalty=-1.0,
        step_penalty=-0.01,
        progress_reward=0.05,
        stall_penalty=-0.5,
    ):
        super().__init__()

        self.size = size
        self.inner_size = size - 2
        self.capacity = self.inner_size * self.inner_size

        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.step_penalty = step_penalty
        self.progress_reward = progress_reward
        self.stall_penalty = stall_penalty

        self.action_space = spaces.Discrete(len(RELATIVE_ACTIONS))
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(4, self.inner_size, self.inner_size),
                    dtype=np.float32,
                ),
                "features": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(12,),
                    dtype=np.float32,
                ),
            }
        )

        self.snake_map = None
        self.snake = None
        self.steps = 0
        self.steps_since_food = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.snake_map = Map(self.size, self.size)
        self.snake = Snake(self.snake_map)
        self.snake_map.create_rand_food()
        self.steps = 0
        self.steps_since_food = 0

        return self._get_obs(), self._get_info(event="reset")

    def step(self, action):
        direction = self._resolve_action(action)
        prev_food_distance = self._food_distance()
        len_before = self.snake.len()

        self.snake.move(direction)
        if not self.snake_map.has_food():
            self.snake_map.create_rand_food()

        self.steps += 1
        self.steps_since_food += 1

        got_food = self.snake.len() > len_before
        if got_food:
            self.steps_since_food = 0

        terminated = self.snake.dead or self.snake_map.is_full()
        truncated = False
        event = "move"

        reward = self.step_penalty
        if got_food:
            reward += self.food_reward
            event = "food"
        elif self.snake.dead:
            reward += self.death_penalty
            event = "death"
        else:
            reward += self.progress_reward * (prev_food_distance - self._food_distance())

        if self.snake_map.is_full():
            event = "full"

        if not terminated and self.steps_since_food >= self._stall_limit():
            truncated = True
            reward += self.stall_penalty
            event = "stall"

        return self._get_obs(), float(reward), terminated, truncated, self._get_info(event=event)

    def close(self):
        return None

    def _resolve_action(self, action):
        heading = self.snake.direc
        relative_action = RELATIVE_ACTIONS[int(action)]
        if relative_action == "left":
            return TURN_LEFT[heading]
        if relative_action == "right":
            return TURN_RIGHT[heading]
        return heading

    def _stall_limit(self):
        return max(self.capacity * 2, self.snake.len() * self.inner_size * 2)

    def _food_distance(self):
        if not self.snake_map.has_food():
            return 0
        return Pos.manhattan_dist(self.snake.head(), self.snake_map.food)

    def _rotate_grid(self, grid):
        return np.rot90(grid, k=ROTATIONS[self.snake.direc]).copy()

    def _rotate_position(self, pos):
        row = pos.x - 1
        col = pos.y - 1
        n = self.inner_size
        k = ROTATIONS[self.snake.direc]

        if k == 0:
            return row, col
        if k == 1:
            return n - 1 - col, row
        if k == 2:
            return n - 1 - row, n - 1 - col
        return col, n - 1 - row

    def _ray_distance(self, direction):
        pos = self.snake.head().adj(direction)
        distance = 0
        while self.snake_map.is_safe(pos):
            distance += 1
            pos = pos.adj(direction)
        return distance

    def _relative_directions(self):
        heading = self.snake.direc
        return (
            TURN_LEFT[heading],
            heading,
            TURN_RIGHT[heading],
        )

    def _board_channels(self):
        channels = np.zeros((4, self.inner_size, self.inner_size), dtype=np.float32)
        for row in range(1, self.size - 1):
            for col in range(1, self.size - 1):
                point_type = self.snake_map.point(Pos(row, col)).type
                idx = (row - 1, col - 1)
                if point_type == PointType.EMPTY:
                    channels[3][idx] = 1.0
                elif point_type == PointType.FOOD:
                    channels[2][idx] = 1.0
                elif point_type in (PointType.HEAD_L, PointType.HEAD_U, PointType.HEAD_R, PointType.HEAD_D):
                    channels[0][idx] = 1.0
                else:
                    channels[1][idx] = 1.0

        return np.stack([self._rotate_grid(channel) for channel in channels], axis=0)

    def _feature_vector(self):
        left, forward, right = self._relative_directions()
        head_row, head_col = self._rotate_position(self.snake.head())
        tail_row, tail_col = self._rotate_position(self.snake.tail())
        norm = max(1, self.inner_size - 1)

        if self.snake_map.has_food():
            food_row, food_col = self._rotate_position(self.snake_map.food)
            food_forward = (head_row - food_row) / norm
            food_right = (food_col - head_col) / norm
        else:
            food_forward = 0.0
            food_right = 0.0

        tail_forward = (head_row - tail_row) / norm
        tail_right = (tail_col - head_col) / norm

        left_distance = self._ray_distance(left) / self.inner_size
        forward_distance = self._ray_distance(forward) / self.inner_size
        right_distance = self._ray_distance(right) / self.inner_size

        features = np.array(
            [
                1.0 if left_distance == 0.0 else 0.0,
                1.0 if forward_distance == 0.0 else 0.0,
                1.0 if right_distance == 0.0 else 0.0,
                left_distance,
                forward_distance,
                right_distance,
                food_forward,
                food_right,
                tail_forward,
                tail_right,
                (self.snake.len() - 2) / max(1, self.capacity - 2),
                min(1.0, self.steps_since_food / self._stall_limit()),
            ],
            dtype=np.float32,
        )
        return features

    def _get_obs(self):
        return {
            "board": self._board_channels(),
            "features": self._feature_vector(),
        }

    def _get_info(self, event):
        return {
            "snake_len": self.snake.len(),
            "steps": self.steps,
            "steps_since_food": self.steps_since_food,
            "stall_limit": self._stall_limit(),
            "event": event,
        }
