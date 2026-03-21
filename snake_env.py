import random

import gymnasium as gym
import numpy as np
from snake.base import Direc, Map, Pos, Snake
from snake.solver.greedy import GreedySolver


RELATIVE_ACTIONS = {
    0: "left",
    1: "straight",
    2: "right",
}

LEFT_OF = {
    Direc.LEFT: Direc.DOWN,
    Direc.UP: Direc.LEFT,
    Direc.RIGHT: Direc.UP,
    Direc.DOWN: Direc.RIGHT,
}

RIGHT_OF = {
    Direc.LEFT: Direc.UP,
    Direc.UP: Direc.RIGHT,
    Direc.RIGHT: Direc.DOWN,
    Direc.DOWN: Direc.LEFT,
}

HEADING_INDEX = {
    Direc.LEFT: 0,
    Direc.UP: 1,
    Direc.RIGHT: 2,
    Direc.DOWN: 3,
}


class SnakeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, size=10, step_penalty=0.01, max_idle_steps=None, max_episode_steps=None):
        super().__init__()
        self.size = size
        self.playable_size = size - 2
        self.capacity = self.playable_size * self.playable_size
        self.max_food_distance = float(2 * (self.playable_size - 1))
        self.step_penalty = step_penalty
        self.max_idle_steps = max_idle_steps or max(40, 2 * self.capacity)
        self.max_episode_steps = max_episode_steps or (12 * self.capacity)

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(277,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)

        self.snake_map = None
        self.snake = None
        self.steps = 0
        self.steps_since_food = 0
        self.last_food_distance = 0.0

    def _relative_direction(self, action):
        move = RELATIVE_ACTIONS[int(action)]
        if move == "left":
            return LEFT_OF[self.snake.direc]
        if move == "right":
            return RIGHT_OF[self.snake.direc]
        return self.snake.direc

    def _danger(self, direction):
        next_pos = self.snake.head().adj(direction)
        return 0.0 if self.snake_map.is_safe(next_pos) else 1.0

    def _clearance(self, direction):
        distance = 0
        pos = self.snake.head().adj(direction)
        while self.snake_map.is_safe(pos):
            distance += 1
            pos = pos.adj(direction)
        return distance / self.playable_size

    def _relative_vector(self, target):
        if target is None:
            return 0.0, 0.0
        head = self.snake.head()
        dx = target.x - head.x
        dy = target.y - head.y

        if self.snake.direc == Direc.UP:
            forward = -dx
            left = -dy
        elif self.snake.direc == Direc.DOWN:
            forward = dx
            left = dy
        elif self.snake.direc == Direc.LEFT:
            forward = -dy
            left = dx
        else:
            forward = dy
            left = -dx

        return forward / self.max_food_distance, left / self.max_food_distance

    def _food_vector(self):
        return self._relative_vector(self.snake_map.food)

    def _tail_vector(self):
        return self._relative_vector(self.snake.tail())

    def _heading_one_hot(self):
        heading = np.zeros(4, dtype=np.float32)
        heading[HEADING_INDEX[self.snake.direc]] = 1.0
        return heading

    def _food_distance(self):
        if self.snake_map.food is None:
            return 0
        return Pos.manhattan_dist(self.snake.head(), self.snake_map.food)

    def _board_features(self):
        body = np.zeros((self.playable_size, self.playable_size), dtype=np.float32)
        head = np.zeros((self.playable_size, self.playable_size), dtype=np.float32)
        tail = np.zeros((self.playable_size, self.playable_size), dtype=np.float32)
        food = np.zeros((self.playable_size, self.playable_size), dtype=np.float32)

        for index, pos in enumerate(self.snake.bodies):
            x = pos.x - 1
            y = pos.y - 1
            if not (0 <= x < self.playable_size and 0 <= y < self.playable_size):
                continue
            if index == 0:
                head[x, y] = 1.0
            elif index == self.snake.len() - 1:
                tail[x, y] = 1.0
            else:
                body[x, y] = 1.0

        if self.snake_map.food is not None:
            food_pos = self.snake_map.food
            x = food_pos.x - 1
            y = food_pos.y - 1
            if 0 <= x < self.playable_size and 0 <= y < self.playable_size:
                food[x, y] = 1.0

        return np.concatenate(
            (
                body.reshape(-1),
                head.reshape(-1),
                tail.reshape(-1),
                food.reshape(-1),
            )
        )

    def _greedy_hint(self):
        hint = np.zeros(3, dtype=np.float32)
        if self.snake.dead or self.snake_map.food is None:
            return hint
        direction = GreedySolver(self.snake).next_direc()
        if direction == LEFT_OF[self.snake.direc]:
            hint[0] = 1.0
        elif direction == self.snake.direc:
            hint[1] = 1.0
        elif direction == RIGHT_OF[self.snake.direc]:
            hint[2] = 1.0
        return hint

    def _get_obs(self):
        left = LEFT_OF[self.snake.direc]
        right = RIGHT_OF[self.snake.direc]
        food_forward, food_left = self._food_vector()
        tail_forward, tail_left = self._tail_vector()
        return np.array(
            [
                self._danger(left),
                self._danger(self.snake.direc),
                self._danger(right),
                self._clearance(left),
                self._clearance(self.snake.direc),
                self._clearance(right),
                food_forward,
                food_left,
                tail_forward,
                tail_left,
                *self._heading_one_hot(),
                self._food_distance() / self.max_food_distance,
                self.snake.len() / self.capacity,
                self.steps_since_food / self.max_idle_steps,
                (self.capacity - self.snake.len()) / self.capacity,
                *self._greedy_hint(),
                *self._board_features(),
            ],
            dtype=np.float32,
        )

    def _score(self):
        if self.steps == 0:
            return 0.0
        return float((self.snake.len() ** 2) / self.steps)

    def _get_info(self):
        return {
            "snake_len": self.snake.len(),
            "steps": self.steps,
            "score": self._score(),
        }

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
        self.last_food_distance = float(self._food_distance())

        return self._get_obs(), self._get_info()

    def step(self, action):
        direction = self._relative_direction(action)
        len_before = self.snake.len()
        prev_distance = self.last_food_distance

        self.snake.move(direction)
        if not self.snake_map.has_food():
            self.snake_map.create_rand_food()

        self.steps += 1
        self.steps_since_food += 1

        terminated = self.snake.dead or self.snake_map.is_full()
        idle_truncated = self.steps_since_food >= self.max_idle_steps
        limit_truncated = self.steps >= self.max_episode_steps
        truncated = idle_truncated or limit_truncated

        got_food = self.snake.len() > len_before
        reward = -self.step_penalty

        if got_food:
            reward += 1.0 + (self.snake.len() / self.capacity)
            self.steps_since_food = 0
        elif self.snake.dead:
            reward -= 1.0
        else:
            current_distance = float(self._food_distance())
            reward += 0.2 * ((prev_distance - current_distance) / self.max_food_distance)

        if truncated and not terminated:
            reward -= 0.5

        self.last_food_distance = float(self._food_distance())
        return self._get_obs(), reward, terminated, truncated, self._get_info()
