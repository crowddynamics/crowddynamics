from collections import namedtuple, Iterable

import numpy as np
from scipy.stats import truncnorm


class Params:
    """
    Generates random parameters for simulations and testing.
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
        values = np.array(tn.rvs(size) * scale / 3.0 + loc)
        return values

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

    @staticmethod
    def to_col_vector(arr):
        return arr.reshape((len(arr), 1))

    def random_position(self, position, radius, x_dims=None, y_dims=None,
                        walls=None):
        """
        Generate uniformly distributed random positions inside x_dims and y_dims for
        the agents without overlapping each others or the walls with Monte Carlo
        method.
        """
        # TODO: define area more accurately
        # TODO: check if area can be filled
        if not isinstance(walls, Iterable):
            walls = (walls,)
        walls = tuple(filter(None, walls))

        i = 0

        if x_dims is None:
            x_dims = self.x
        if y_dims is None:
            y_dims = self.y

        iterations = 0
        max_iterations = 100 * len(position)
        while i < len(position):
            if iterations >= max_iterations:
                raise Exception("Iteration limit if {} reached.".format(
                    max_iterations))
            iterations += 1

            # Random uniform position inside x and y dimensions
            pos = np.zeros(2)

            pos[0] = np.random.uniform(*x_dims)
            pos[1] = np.random.uniform(*y_dims)

            if isinstance(radius, np.ndarray):
                rad = radius[i]
                radii = radius[:i]
            else:
                rad = radius
                radii = radius

            # Test overlapping with other agents
            if i > 0:
                d = pos - position[:i]
                d = np.hypot(d[:, 0], d[:, 1]) - (rad + radii)
                cond = np.all(d > 0)
                if not cond:
                    continue

            # Test overlapping with walls
            cond = 1
            for wall in walls:
                for j in range(wall.size):
                    d = wall.distance(j, pos) - rad
                    cond *= d > 0
            if not cond:
                continue

            position[i, :] = pos
            i += 1

    def agent(self, size):
        """Arguments for constructing agent."""
        mass = self.truncnorm(loc=70.0, scale=10.0, size=size)
        radius = self.truncnorm(loc=0.255, scale=0.035, size=size)
        goal_velocity = 5.0 * np.ones(size)
        goal_reached = np.zeros(size, dtype=np.bool_)
        mass = self.to_col_vector(mass)
        radius = self.to_col_vector(radius)
        goal_velocity = self.to_col_vector(goal_velocity)
        return size, mass, radius, goal_velocity, goal_reached

    def round_wall(self, size, r_min, r_max):
        """Arguments for constructing round wall."""
        return np.stack((np.random.uniform(self.x.min, self.x.max, size),
                         np.random.uniform(self.y.min, self.y.max, size),
                         np.random.uniform(r_min, r_max, size=size)), axis=1)

    def linear_wall(self, size):
        """Arguments for constructing linear wall."""
        args = zip(self.random_2D_coordinates(size),
                   self.random_2D_coordinates(size))
        return np.array(tuple(args))
