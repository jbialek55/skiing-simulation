import numpy as np
from mesa import Model
from mesa.space import MultiGrid


class SkiModel(Model):
    def __init__(self, width=100, height=200):
        super().__init__()
        self.width = width
        self.height = height

        # siatka
        self.grid = MultiGrid(width, height, torus=False)
        # wysokość
        self.elevation_map = np.zeros((width, height))
        # prawdopodobieństwa
        self.safety_map = np.ones((width, height))
        
        self._generate_slope_profile()
        self._setup_obstacles()

    def _generate_slope_profile(self):
        pass

    def _setup_obstacles(self):
        pass

    def step(self):
        pass
