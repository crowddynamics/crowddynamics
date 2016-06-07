from collections import OrderedDict, Iterable
from copy import deepcopy

import numpy as np
from numba.types import UniTuple
from numba import float64, int64, boolean
from numba import jitclass, generated_jit, types

from src.core.functions import normalize_vec, normalize


@generated_jit(nopython=True)
def get_scalar_or_array(value, i):
    if isinstance(value, types.Float):
        return lambda value, i: value
    elif isinstance(value, types.Array):
        return lambda value, i: value[i, 0]
    else:
        raise ValueError()


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

agent_attr_names = [key for key in spec_agent.keys()]


class Agent(object):
    """
    Structure for agent parameters and variables.
    """

    def __init__(self, size, mass, radius, goal_velocity, goal_reached):
        self.size = size
        self.shape = (size, 2)

        # Scalars or vectors of shape=(size, 1)
        # TODO: Elliptical Agents, Orientation, Major- & Minor axis
        # TODO: Three circles representation
        # TODO: Collection of average human dimensions and properties
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
        self.herding_flag = False                       #
        self.herding_tendency = np.zeros(self.size)     #
        self.neighbor_direction = np.zeros(self.shape)  #
        self.neighbors = np.zeros(self.size)            #

        # TODO: Path finding
        # https://en.wikipedia.org/wiki/Pathfinding

    def reset_force(self):
        self.force *= 0

    def reset_force_debug(self):
        self.force_adjust *= 0
        self.force_agent *= 0
        self.force_wall *= 0

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
    goal_velocity can be scalar or array.
    """
    _spec_agent = deepcopy(spec_agent)

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
        _spec_agent[key] = t
        return value

    # Init spec_agent
    mass = spec('mass', mass)
    radius = spec('radius', radius)
    goal_velocity = spec('goal_velocity', goal_velocity)
    goal_reached = np.zeros(size, dtype=np.bool_)

    # Return jitclass of Agents
    agent = jitclass(_spec_agent)(Agent)
    return agent(size, mass, radius, goal_velocity, goal_reached)

