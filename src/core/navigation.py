import numpy as np
import skfmm

from .vector2d import angle_nx2


class ExitSelection:
    """Exit selection policy."""
    def __init__(self, agent, exits):
        pass


class Navigation:
    """Target direction"""
    def __init__(self, agent, domain, obstacles, exits):
        self.agent = agent
        self.domain = domain
        self.obstacles = obstacles
        self.exits = exits

    def distance_map(self, n=201, dx=1e-2):
        # Discretize the domain
        x = np.linspace(self.domain.x[0], self.domain.x[1], n)
        y = np.linspace(self.domain.y[0], self.domain.y[0], n)
        X, Y = np.meshgrid(x, y)

        # Set contour.
        phi = -1 * np.ones_like(X)

        # Exits. Zero contour defines the exit.
        # phi[] = 1

        # Obstacles are defined by masked values.
        mask = None

        phi = np.ma.MaskedArray(phi, mask)
        d_map = skfmm.distance(phi, dx=dx)

        return

    def static_potential(self):
        pass

    def dynamic_potential(self):
        pass

    def update(self):
        # self.agent.target_direction = None
        pass


class Orientation:
    """Target orientation"""
    def __init__(self, agent):
        self.agent = agent

    def update(self):
        self.agent.target_angle[:] = angle_nx2(self.agent.target_direction)
