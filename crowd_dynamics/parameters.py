from collections import namedtuple, Iterable

import numpy as np
from scipy.stats import truncnorm


# column = namedtuple("column", ("adult", "male", "female", "child", "eldery"))
# radius, dr, torso, shoulder, shoulder distance, walking speed, dv
# TODO: Body mass and rotational moment values table
index = namedtuple(
    "index", ("r", "dr", "k_t", "k_s", "k_ts", "v", "dv", "mass", "mass_scale"))
body_types = dict(
    adult=index(0.255, 0.035, 0.5882, 0.3725, 0.6275, 1.25, 0.30, 75, 7),
    male=index(0.270, 0.020, 0.5926, 0.3704, 0.6296, 1.35, 0.20, 82, 10),
    female=index(0.240, 0.020, 0.5833, 0.3750, 0.6250, 1.15, 0.20, 67, 5),
    child=index(0.210, 0.015, 0.5714, 0.3333, 0.6667, 0.90, 0.30, 57, 5),
    eldery=index(0.250, 0.020, 0.6000, 0.3600, 0.6400, 0.80, 0.30, None, None),
)


class Parameters:
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
        std = 3.0
        tn = truncnorm(-std, std)
        values = np.array(tn.rvs(size) * scale + loc)
        return values

    def random_2d_coordinates(self, size):
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

    def random_position(self, position, radius, x_dims=None, y_dims=None,
                        walls=None):
        """
        Generate uniformly distributed random positions inside x_dims and y_dims
        for the agents without overlapping each others or the walls with Monte
        Carlo method.
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

    def agent(self, size, three_circles_flag=True, body_type="adult"):
        """Arguments for constructing agent."""
        body = body_types[body_type]
        mass = self.truncnorm(loc=body.mass, scale=body.mass_scale, size=size)
        radius = self.truncnorm(loc=body.r, scale=body.dr, size=size)
        r_t = body.k_t * radius
        r_s = body.k_s * radius
        r_ts = body.k_ts * radius
        inertia_rot_scale = 4.0 / (80.0 * 0.255 ** 2)
        inertia_rot = inertia_rot_scale * mass * radius ** 2  # I = mr^2
        goal_velocity = 5.0 * np.ones(size)
        target_angular_velocity = 4 * np.pi * np.ones(size)

        return size, mass, radius, r_t, r_s, r_ts, inertia_rot, goal_velocity, \
               target_angular_velocity, three_circles_flag

    def round_wall(self, size, r_min, r_max):
        """Arguments for constructing round wall."""
        return np.stack((np.random.uniform(self.x.min, self.x.max, size),
                         np.random.uniform(self.y.min, self.y.max, size),
                         np.random.uniform(r_min, r_max, size=size)), axis=1)

    def linear_wall(self, size):
        """Arguments for constructing linear wall."""
        args = zip(self.random_2d_coordinates(size),
                   self.random_2d_coordinates(size))
        return np.array(tuple(args))
