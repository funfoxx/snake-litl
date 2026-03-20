from __future__ import annotations

from collections import defaultdict, deque

import gymnasium as gym
import numpy as np

from snake.base import Direc, Map, Pos, Snake

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

        # 4 global progress features + 8 successor-state features for each
        # relative action (left, straight, right).
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(28,),
            dtype=np.float32,
        )
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

    def _max_steps_without_food(self) -> int:
        return max(self.capacity, self.snake.len() * 12)

    def _is_inside(self, pos: Pos) -> bool:
        return (
            0 < pos.x < self.snake_map.num_rows - 1
            and 0 < pos.y < self.snake_map.num_cols - 1
        )

    def _normalize_ratio(self, value: float) -> float:
        return float(np.clip((2.0 * value) - 1.0, -1.0, 1.0))

    def _normalize_distance(self, distance: int | None) -> float:
        if distance is None:
            return -1.0
        return float(np.clip(1.0 - (2.0 * distance / max(1, self.capacity)), -1.0, 1.0))

    def _transition_bodies(self, direction: Direc) -> tuple[list[Pos], bool]:
        new_head = self.snake.head().adj(direction)
        if not self.snake_map.is_safe(new_head):
            return [], False

        got_food = self.snake_map.food is not None and new_head == self.snake_map.food
        bodies = [new_head, *list(self.snake.bodies)]
        if not got_food:
            bodies.pop()
        return bodies, got_food

    def _reachable_count(
        self,
        start: Pos,
        occupied: set[Pos],
        goal: Pos | None = None,
    ) -> int:
        queue = deque([start])
        visited = {start}
        count = 0

        while queue:
            current = queue.popleft()
            count += 1
            for next_pos in current.all_adj():
                if next_pos in visited or not self._is_inside(next_pos):
                    continue
                if next_pos in occupied and next_pos != goal:
                    continue
                visited.add(next_pos)
                queue.append(next_pos)

        return count

    def _shortest_path_length(
        self,
        start: Pos,
        goal: Pos,
        occupied: set[Pos],
        goal_is_passable: bool = False,
    ) -> int | None:
        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current, dist = queue.popleft()
            if current == goal:
                return dist

            for next_pos in current.all_adj():
                if next_pos in visited or not self._is_inside(next_pos):
                    continue
                if next_pos in occupied and not (goal_is_passable and next_pos == goal):
                    continue
                visited.add(next_pos)
                queue.append((next_pos, dist + 1))

        return None

    def _branching_factor(self, pos: Pos, occupied: set[Pos]) -> int:
        branches = 0
        for next_pos in pos.all_adj():
            if self._is_inside(next_pos) and next_pos not in occupied:
                branches += 1
        return branches

    def _action_features(self, action: int) -> list[float]:
        direction = self._action_to_direction(action)
        new_bodies, got_food = self._transition_bodies(direction)
        if not new_bodies:
            return [-1.0] * 8

        new_head = new_bodies[0]
        occupied = set(new_bodies)
        distance_before = Pos.manhattan_dist(self.snake.head(), self.snake_map.food)

        if got_food and len(new_bodies) >= self.capacity:
            return [1.0] * 8

        if got_food:
            food_reachable = 1.0
            food_distance = 1.0
            progress = 1.0
        else:
            shortest_path = self._shortest_path_length(
                new_head,
                self.snake_map.food,
                occupied,
            )
            food_reachable = 1.0 if shortest_path is not None else -1.0
            food_distance = self._normalize_distance(shortest_path)

            distance_after = Pos.manhattan_dist(new_head, self.snake_map.food)
            if distance_after < distance_before:
                progress = 1.0
            elif distance_after > distance_before:
                progress = -1.0
            else:
                progress = 0.0

        reachable_ratio = self._normalize_ratio(
            self._reachable_count(new_head, occupied) / float(self.capacity)
        )
        branching_ratio = self._normalize_ratio(
            self._branching_factor(new_head, occupied) / 4.0
        )

        tail_reachable = 1.0
        if len(new_bodies) > 2:
            tail_reachable = (
                1.0
                if self._shortest_path_length(
                    new_head,
                    new_bodies[-1],
                    occupied,
                    goal_is_passable=True,
                )
                is not None
                else -1.0
            )

        safe_moves = 0
        for next_direction in (
            TURN_LEFT[direction],
            direction,
            TURN_RIGHT[direction],
        ):
            next_pos = new_head.adj(next_direction)
            if self._is_inside(next_pos) and next_pos not in occupied:
                safe_moves += 1
        safe_moves_ratio = self._normalize_ratio(safe_moves / 3.0)

        return [
            1.0,
            food_reachable,
            food_distance,
            reachable_ratio,
            branching_ratio,
            tail_reachable,
            safe_moves_ratio,
            progress,
        ]

    def _get_obs(self):
        forward_food, lateral_food = self._food_vector()
        length_ratio = (self.snake.len() - 2) / max(1, self.capacity - 2)
        starvation_ratio = 1.0 - (
            self.steps_since_food / float(self._max_steps_without_food())
        )

        action_features = []
        for action in range(3):
            action_features.extend(self._action_features(action))

        return np.asarray(
            [
                forward_food,
                lateral_food,
                float(np.clip(length_ratio, 0.0, 1.0)),
                float(np.clip(starvation_ratio, -1.0, 1.0)),
            ]
            + action_features,
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
        distance_before = Pos.manhattan_dist(head_before, self.snake_map.food)
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
