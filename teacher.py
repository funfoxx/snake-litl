from snake.solver.greedy import GreedySolver
from snake.solver.path import PathSolver


class TeacherSolver:
    def __init__(self, snake, mode="greedy", aggressive_food_len=12):
        self.mode = mode
        self.aggressive_food_len = aggressive_food_len
        self._greedy = GreedySolver(snake)
        self._path = PathSolver(snake)

    @property
    def snake(self):
        return self._greedy.snake

    @snake.setter
    def snake(self, snake):
        self._greedy.snake = snake
        self._path.snake = snake

    def next_direc(self):
        if self.mode == "greedy":
            return self._greedy.next_direc()

        if self.mode == "aggressive_food":
            if self.snake.len() < self.aggressive_food_len:
                path = self._path.shortest_path_to_food()
                if path:
                    return path[0]
            return self._greedy.next_direc()

        raise ValueError(f"Unsupported teacher mode: {self.mode}")
