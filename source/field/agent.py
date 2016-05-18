from collections import OrderedDict

import numpy as np
from numba import jitclass, float64, int64, generated_jit, types


@generated_jit(nopython=True)
def get_radius_gen(radius, i):
    if isinstance(radius, types.Float):
        return lambda radius, i: radius
    elif isinstance(radius, types.Array):
        return lambda radius, i: radius[i, 0]
    else:
        raise ValueError()


class Agent(object):
    """
    Structure for agent parameters and variables.
    """

    def __init__(self, mass, radius, position, velocity, goal_velocity,
                 goal_direction):
        # Scalars or vectors of shape=(size, 1)
        self.mass = mass
        self.radius = radius
        self.goal_velocity = goal_velocity

        # Vectors of shape=(size, 2)
        self.position = position
        self.velocity = velocity
        self.goal_direction = goal_direction

        # Arrays can be iterated over range(size)
        self.size = len(self.position)  # Number of rows in position

        # TODO: Vectors for gathering forces for debugging
        # self.force = np.zeros(self.shape)

    @property
    def shape(self):
        return self.position.shape

    def get_radius(self, i):
        return get_radius_gen(self.radius, i)

    # TODO: Target direction


def agent_struct(mass,
                 radius,
                 position,
                 velocity,
                 goal_velocity,
                 goal_direction):
    """
    Makes jitclass from agents. Handles spec definition so that mass, radius and
    goal_velocity can be scalar of array.
    """
    spec_agent = OrderedDict(
        mass=float64,
        radius=float64,
        goal_velocity=float64,
        position=float64[:, :],
        velocity=float64[:, :],
        goal_direction=float64[:, :],
        size=int64,
    )

    def spec(key, value):
        if isinstance(value, (int, float)):
            value = float(value)
            t = float64
        elif isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)
            value = value.reshape((len(value), 1))
            t = float64[:, :]
        else:
            raise ValueError("Wrong type.")
        spec_agent[key] = t
        return value

    # Init spec_agent
    mass = spec('mass', mass)
    radius = spec('radius', radius)
    goal_velocity = spec('goal_velocity', goal_velocity)
    # Jitclass of Agents
    agent = jitclass(spec_agent)(Agent)
    # Set parameters
    args = (mass, radius, position, velocity, goal_velocity, goal_direction)
    return agent(*args)


def initial_position(amount, x_dims, y_dims, radius, linear_wall=None):
    """
    Populate the positions of the agents in to the field so that they don't
    overlap each others or the walls.

    Monte Carlo method.
    """
    i = 0
    position = np.zeros((amount, 2))
    while i < amount:
        # Random uniform position inside x and y dimensions
        pos = np.zeros(2)
        pos[0] = np.random.uniform(x_dims[0], x_dims[1])
        pos[1] = np.random.uniform(y_dims[0], y_dims[1])
        others = position[:i]

        if isinstance(radius, np.ndarray):
            rad = radius[i]
            radii = radius[:i]
        else:
            rad = radius
            radii = radius

        # Test overlapping with other agents
        if len(others) > 0:
            d = pos - others
            d = np.hypot(d[:, 0], d[:, 1]) - (rad + radii)
            cond = np.all(d > 0)
            if not cond:
                continue

        # Test overlapping with walls
        if linear_wall is not None:
            cond = 1
            for j in range(linear_wall.size):
                d = linear_wall.distance(j, pos) - rad
                cond *= d > 0
            if not cond:
                continue

        position[i, :] = pos
        i += 1
    return position


def initial_velocity(amount):
    """
    Set velocities.
    """
    orientation = np.random.uniform(0, 2 * np.pi, amount)
    velocity = np.stack((np.cos(orientation), np.sin(orientation)), axis=1)
    return velocity
