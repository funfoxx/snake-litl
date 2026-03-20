import numpy as np
import gymnasium as gym
from snake.base import Direc, Map, PointType, Pos, Snake

ACTIONS = {
    0: Direc.LEFT,
    1: Direc.UP,
    2: Direc.RIGHT,
    3: Direc.DOWN
}

class SnakeEnv(gym.Env):

    def __init__(self, size=10):
        self.size = size

        self.observation_space = gym.spaces.Box(0, 4, shape=(self.size, self.size))
        self.action_space = gym.spaces.Discrete(4)

    def _get_obs(self):
        obs = np.zeros((self.size, self.size), dtype=np.int8)
        for i in range(self.size):
            for j in range(self.size):
                p = self.snake_map.point(Pos(i, j)).type
                if p == PointType.EMPTY:
                    obs[i, j] = 0
                elif p == PointType.WALL: 
                    obs[i, j] = 1 
                elif p == PointType.FOOD: 
                    obs[i, j] = 2
                elif p == PointType.HEAD_L or p == PointType.HEAD_U or p == PointType.HEAD_R or p == PointType.HEAD_D:
                    obs[i, j] = 3 
                # body
                else:
                    obs[i, j] = 4

        return obs
    
    def _get_info(self):
        return {
            "snake_len": self.snake.len(),
            "steps": self.steps
        }
    
    def reset(self, seed=None):
        super().reset(seed=seed)

        self.snake_map = Map(self.size, self.size)
        self.snake = Snake(self.snake_map)
        self.snake_map.create_rand_food()
        self.steps = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        direction = ACTIONS[int(action)]

        len_before = self.snake.len()

        self.snake.move(direction)
        if not self.snake_map.has_food():
            self.snake_map.create_rand_food() 
        self.steps += 1

        terminated = self.snake.dead or self.snake_map.is_full()
        truncated = self.steps > 10000

        got_food = self.snake.len() > len_before
        if got_food:
            reward = 1.0
        elif self.snake.dead:
            reward = -1.0
        else:
            reward = 0.0 
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info