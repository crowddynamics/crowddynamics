from collections import OrderedDict, Iterable
from copy import deepcopy

import numpy as np
import numba
from numba.types import UniTuple
from numba import float64, int64, boolean
from numba import jitclass, generated_jit, types

from src.core.functions import normalize_vec, normalize


spec_agent = OrderedDict(
    size=int64,
    shape=UniTuple(int64, 2),
    mass=float64[:, :],
    radius=float64[:, :],
    goal_velocity=float64[:, :],
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


@numba.jitclass(spec_agent)
class Agent(object):
    """
    Structure for agent parameters and variables.
    """

    def __init__(self, size, mass, radius, goal_velocity, goal_reached):
        """

        :param size: Integer. Size of the arrays.
        :param mass: Array of masses of agents.
        :param radius: Array of radii of agents.
        :param goal_velocity: Array of goal_velocities of agents.
        :param goal_reached: Array of boolean values of agents that have reached
        their goals. Should be initialized to all false.
        """
        self.size = size
        self.shape = (size, 2)

        # Scalars or vectors of shape=(size, 1)
        # TODO: Elliptical Agents, Orientation, Major- & Minor axis
        # TODO: Three circles representation
        # TODO: Collection of average human dimensions and properties
        self.radius = radius

        self.mass = mass
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

    def set_goal_direction(self, goal):
        mask = self.goal_reached ^ True
        if np.sum(mask):
            self.goal_direction[mask] = normalize_vec(goal - self.position[mask])
