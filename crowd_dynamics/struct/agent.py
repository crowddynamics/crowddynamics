from collections import OrderedDict

import numba
import numpy as np
from numba import float64, int64, boolean
from numba.types import UniTuple

from ..core.vector2d import normalize_vec, normalize


spec_agent = OrderedDict(
    size=int64,
    shape=UniTuple(int64, 2),

    three_circles_flag=boolean,
    orientable_flag=boolean,
    herding_flag=boolean,

    mass=float64[:, :],
    radius=float64[:],
    radius_torso=float64[:],
    radius_shoulder=float64[:],
    radius_torso_shoulder=float64[:],
    goal_velocity=float64[:, :],

    inertia_rot=float64[:],
    angle=float64[:],
    angular_velocity=float64[:],
    target_angle=float64[:],
    target_angular_velocity=float64[:],
    torque=float64[:],

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

    def __init__(self,
                 size,
                 mass,
                 radius,
                 radius_torso,
                 radius_shoulder,
                 radius_torso_shoulder,
                 inertia_rot,
                 goal_velocity,
                 goal_reached):
        """

        :param size: Integer. Size of the arrays.
        :param mass: Array of masses of agents.
        :param radius: Array of radii of agents.
        :param goal_velocity: Array of goal_velocities of agents.
        :param goal_reached: Array of boolean values of agents that have reached
        their goals. Should be initialized to all false.
        """

        # TODO: Three circles, Orientation
        # TODO: Collection of average human dimensions and properties
        # TODO: Path finding
        # TODO: Goal reached? Handle reached goals.
        # TODO: Gathering other forces for debugging and plotting
        # TODO: Not see through walls?

        # Array properties
        self.size = size
        self.shape = (size, 2)

        # Flags - Which features are active.
        self.three_circles_flag = False
        self.orientable_flag = self.three_circles_flag
        self.herding_flag = False

        # Agent properties
        self.radius = radius
        self.radius_torso = radius_torso
        self.radius_shoulder = radius_shoulder
        self.radius_torso_shoulder = radius_torso_shoulder

        self.mass = mass.reshape(size, 1)
        self.inertia_rot = inertia_rot

        # Rotational movement
        self.angle = np.zeros(self.size)
        self.angular_velocity = np.zeros(self.size)
        self.target_angle = np.zeros(self.size)
        self.target_angular_velocity = np.zeros(self.size)
        self.torque = np.zeros(self.size)

        # Movement along x and y axis
        self.position = np.zeros(self.shape)
        self.velocity = np.zeros(self.shape)
        self.goal_direction = np.zeros(self.shape)
        self.goal_velocity = goal_velocity.reshape(size, 1)
        self.target_direction = np.zeros(self.shape)
        self.force = np.zeros(self.shape)

        # self.goal_position = np.zeros(self.shape)
        self.goal_reached = goal_reached

        self.force_adjust = np.zeros(self.shape)
        self.force_agent = np.zeros(self.shape)
        self.force_wall = np.zeros(self.shape)

        # Distances for reacting to other objects
        self.sight_soc = 7.0
        self.sight_wall = 7.0
        self.sight_herding = 20.0

        # Herding
        self.herding_tendency = np.zeros(self.size)
        self.neighbor_direction = np.zeros(self.shape)
        self.neighbors = np.zeros(self.size)

    def reset(self):
        self.force *= 0
        self.torque *= 0
        self.force_adjust *= 0
        self.force_agent *= 0
        self.force_wall *= 0

    def reset_herding(self):
        self.neighbor_direction *= 0
        self.neighbors *= 0

    def goal_to_target_direction(self):
        """Modifies target direction from goal direction."""
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

    def velocity_to_target_angle(self):
        # TODO: If velocity (0, 0)
        self.target_angle = np.arctan2(self.velocity[:, 0], self.velocity[:, 1])

    def set_goal_direction(self, goal):
        """Update goal direction for agent that have not reached their goals"""
        mask = self.goal_reached ^ True
        if np.sum(mask):
            self.goal_direction[mask] = normalize_vec(goal - self.position[mask])
