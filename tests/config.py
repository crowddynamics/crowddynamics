from collections import namedtuple

import numpy as np
from scipy.stats import truncnorm


class Params:
    """
    Generates parameters for testing algorithms.
    """
    Dim = namedtuple('Dim', ['width', 'height'])
    Lim = namedtuple('Lim', ['min', 'max'])

    def __init__(self, width, height):
        self.dims = self.Dim(width, height)
        self.x = self.Lim(0.0, self.dims.width)
        self.y = self.Lim(0.0, self.dims.height)

    @staticmethod
    def truncnorm(loc, scale, size):
        tn = truncnorm(-3.0, 3.0)
        return tn.rvs(size) * scale / 3.0 + loc

    def random_2D_coordinates(self, size):
        """Random x and y coordinates inside dims."""
        return np.stack((np.random.uniform(self.x.min, self.x.max, size),
                         np.random.uniform(self.y.min, self.y.max, size)),
                        axis=1)

    @staticmethod
    def random_unit_vector(size):
        """Random unit vector."""
        orientation = np.random.uniform(0, 2 * np.pi, size)
        velocity = np.stack((np.cos(orientation), np.sin(orientation)), axis=1)
        return velocity

    def agent(self, size=10):
        """Arguments for constructing agent."""
        mass = self.truncnorm(loc=70.0, scale=10.0, size=size)
        radius = self.truncnorm(loc=0.255, scale=0.035, size=size)
        goal_velocity = 5.0
        return size, mass, radius, goal_velocity

    def round_wall(self, size, radius=0.3):
        """Arguments for constructing round wall."""
        return np.stack((np.random.uniform(self.x.min, self.x.max, size),
                         np.random.uniform(self.y.min, self.y.max, size),
                         np.random.uniform(high=radius, size=size)), axis=1)

    def linear_wall(self, size):
        """Arguments for constructing linear wall."""
        return np.array((self.random_2D_coordinates(size),
                         self.random_2D_coordinates(size)))
