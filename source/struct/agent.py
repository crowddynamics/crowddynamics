from collections import OrderedDict

import numpy as np
from numba import float64, int64
from numba import jitclass, generated_jit, types


@generated_jit(nopython=True)
def get_scalar_or_array(value, i):
    if isinstance(value, types.Float):
        return lambda value, i: value
    elif isinstance(value, types.Array):
        return lambda value, i: value[i, 0]
    else:
        raise ValueError()


class Agent(object):
    """
    Structure for agent parameters and variables.
    """

    def __init__(self, mass, radius, position, velocity, goal_velocity,
                 goal_direction, herding_tendency):
        # Scalars or vectors of shape=(size, 1)
        self.mass = mass
        self.radius = radius
        self.goal_velocity = goal_velocity

        # Vectors of shape=(size, 2)
        self.position = position
        self.velocity = velocity
        self.goal_direction = goal_direction
        self.target_direction = self.goal_direction.copy()
        self.force = np.zeros(self.shape)

        # TODO: Vectors for gathering forces for debugging
        self.force_adjust = np.zeros(self.shape)
        self.force_agent = np.zeros(self.shape)
        self.force_wall = np.zeros(self.shape)

        # Distances for reacting to other objects
        self.sight_soc = 7.0
        self.sight_wall = 7.0
        self.sight_herding = 20.0

        # Herding
        self.herding_flag = 0
        self.herding_tendency = herding_tendency
        self.neighbor_direction = np.zeros(self.shape)
        self.neighbors = np.zeros(self.size)

    @property
    def shape(self):
        return self.position.shape

    @property
    def size(self):
        return self.shape[0]

    def reset_force(self):
        self.force *= 0

    def reset_herding(self):
        self.neighbor_direction *= 0
        self.neighbors *= 0

    def herding_behaviour(self):
        """
        Modifies target direction.
        """
        if self.herding_flag:
            for i in range(self.size):
                p = self.herding_tendency[i]
                neighbor_mean = self.neighbor_direction[i] / self.neighbors[i]
                self.target_direction[i] = (1 - p) * self.goal_direction[i] + \
                                           p * neighbor_mean
            self.reset_herding()
        else:
            self.herding_flag = 1

    def get_radius(self, i):
        """

        :param i: Index.
        :return: Returns radius if scalar or radius[i, 0] if an vector.
        """
        return get_scalar_or_array(self.radius, i)


def agent_struct(mass, radius, position, velocity, goal_velocity,
                 goal_direction, herding_tendency):
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
        target_direction=float64[:, :],
        force=float64[:, :],
        force_adjust=float64[:, :],
        force_agent=float64[:, :],
        force_wall=float64[:, :],
        sight_soc=float64,
        sight_wall=float64,
        sight_herding=float64,
        herding_flag=int64,
        herding_tendency=float64[:],
        neighbor_direction=float64[:, :],
        neighbors=float64[:],
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
    args = (mass, radius, position, velocity, goal_velocity, goal_direction,
            herding_tendency)
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
