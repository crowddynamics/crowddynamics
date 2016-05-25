from collections import OrderedDict

import numpy as np
from numba.types import UniTuple
from numba import float64, int64, boolean
from numba import jitclass, generated_jit, types

from source.core.functions import normalize_vec, normalize


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

    def __init__(self, size, mass, radius, goal_velocity, goal_reached):
        self.size = size
        self.shape = (size, 2)

        # Scalars or vectors of shape=(size, 1)
        # TODO: Elliptical Agents, Orientation, Major- & Minor axis
        self.mass = mass
        self.radius = radius
        self.goal_velocity = goal_velocity

        # Vectors of shape=(size, 2)
        self.position = np.zeros(self.shape)          # Center of mass
        self.velocity = np.zeros(self.shape)          # Current velocity

        # self.goal_position = np.zeros(self.shape)
        self.goal_direction = np.zeros(self.shape)    # Unit vector
        self.target_direction = np.zeros(self.shape)  # Unit vector
        self.force = np.zeros(self.shape)             # Total Force
        # TODO: Goal reached? When target reached do something?
        self.goal_reached = goal_reached

        # TODO: Gathering other forces for debugging and plotting
        self.force_adjust = np.zeros(self.shape)
        self.force_agent = np.zeros(self.shape)
        self.force_wall = np.zeros(self.shape)

        # Distances for reacting to other objects
        # TODO: Not see through walls?
        self.sight_soc = 7.0
        self.sight_wall = 7.0
        self.sight_herding = 20.0

        # Herding
        self.herding_flag = 0                           # 0 | 1 = on | off
        self.herding_tendency = np.zeros(self.size)     #
        self.neighbor_direction = np.zeros(self.shape)  #
        self.neighbors = np.zeros(self.size)            #

        # TODO: Path finding
        # https://en.wikipedia.org/wiki/Pathfinding

    def reset_force(self):
        self.force *= 0

    def reset_herding(self):
        self.neighbor_direction *= 0
        self.neighbors *= 0

    def goal_to_target_direction(self):
        """
        Modifies target direction from goal direction.
        """
        if self.herding_flag:
            # Herding behaviour
            for i in range(self.size):
                p = self.herding_tendency[i]
                mean = self.neighbor_direction[i] / self.neighbors[i]
                self.target_direction[i] = \
                    normalize((1 - p) * self.goal_direction[i] + p * mean)
            self.reset_herding()
        else:
            self.target_direction = self.goal_direction

    def get_radius(self, i):
        """

        :param i: Index.
        :return: Returns radius if scalar or radius[i, 0] if an vector.
        """
        return get_scalar_or_array(self.radius, i)

    def set_goal_direction(self, goal):
        mask = self.goal_reached ^ True
        if np.sum(mask):
            self.goal_direction[mask] = normalize_vec(goal - self.position[mask])


def agent_struct(size, mass, radius, goal_velocity):
    """
    Makes jitclass from agents. Handles spec definition so that mass, radius and
    goal_velocity can be scalar of array.
    """
    spec_agent = OrderedDict(
        size=int64,
        shape=UniTuple(int64, 2),
        mass=float64,
        radius=float64,
        goal_velocity=float64,
        position=float64[:, :],
        velocity=float64[:, :],
        goal_direction=float64[:, :],
        target_direction=float64[:, :],
        force=float64[:, :],
        goal_reached=boolean[:],
        force_adjust=float64[:, :],
        force_agent=float64[:, :],
        force_wall=float64[:, :],
        sight_soc=float64,
        sight_wall=float64,
        sight_herding=float64,
        herding_flag=boolean,
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
    goal_reached = np.zeros(size, dtype=np.bool_)
    # Jitclass of Agents
    agent = jitclass(spec_agent)(Agent)
    return agent(size, mass, radius, goal_velocity, goal_reached)


def random_position(agent, x_dims, y_dims, wall=None):
    """
    Generate uniformly distributed random positions inside x_dims and y_dims for
    the agents without overlapping each others or the walls with Monte Carlo
    method.
    """
    # TODO: agent argument
    # TODO: define area more accurately
    # TODO: check if area can be filled

    i = 0
    iterations = 0
    max_iterations = 100 * agent.size
    while i < agent.size:
        if iterations >= max_iterations:
            raise Exception("Iteration limit if {} reached.".format(
                max_iterations))
        iterations += 1

        # Random uniform position inside x and y dimensions
        pos = np.zeros(2)
        pos[0] = np.random.uniform(x_dims[0], x_dims[1])
        pos[1] = np.random.uniform(y_dims[0], y_dims[1])

        radius = agent.radius
        if isinstance(radius, np.ndarray):
            rad = radius[i]
            radii = radius[:i]
        else:
            rad = radius
            radii = radius

        # Test overlapping with other agents
        if i > 0:
            d = pos - agent.position[:i]
            d = np.hypot(d[:, 0], d[:, 1]) - (rad + radii)
            cond = np.all(d > 0)
            if not cond:
                continue

        # Test overlapping with walls
        if wall is not None:
            cond = 1
            for j in range(wall.size):
                d = wall.distance(j, pos) - rad
                cond *= d > 0
            if not cond:
                continue

        agent.position[i, :] = pos
        i += 1


def random_velocity(amount):
    """
    Set velocities.
    """
    orientation = np.random.uniform(0, 2 * np.pi, amount)
    velocity = np.stack((np.cos(orientation), np.sin(orientation)), axis=1)
    return velocity
