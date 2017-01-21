import numba
import numpy as np
from numba import float64, int64, boolean
from numba.types import UniTuple

from crowddynamics.core.vector2D.vector2D import rotate270

spec_agent = (
    ("size", int64),
    ("shape", UniTuple(int64, 2)),

    ("circular", boolean),
    ("three_circle", boolean),

    ("orientable", boolean),
    ("active", boolean[:]),
    ("goal_reached", boolean[:]),
    ("mass", float64[:, :]),
    ("radius", float64[:]),
    ("r_t", float64[:]),
    ("r_s", float64[:]),
    ("r_ts", float64[:]),

    ("position", float64[:, :]),
    ("velocity", float64[:, :]),
    ("target_velocity", float64[:, :]),
    ("target_direction", float64[:, :]),
    ("force", float64[:, :]),
    ("inertia_rot", float64[:]),
    ("angle", float64[:]),
    ("angular_velocity", float64[:]),
    ("target_angle", float64[:]),
    ("target_angular_velocity", float64[:]),
    ("torque", float64[:]),
    ("position_ls", float64[:, :]),
    ("position_rs", float64[:, :]),
    ("front", float64[:, :]),
)

spec_agent_motion = (
    ("tau_adj", float64),
    ("tau_rot", float64),
    ("k_soc", float64),
    ("tau_0", float64),
    ("mu", float64),
    ("kappa", float64),
    ("damping", float64),
    ("std_rand_force", float64),
    ("std_rand_torque", float64),
    ("f_soc_ij_max", float64),
    ("f_soc_iw_max", float64),
    ("sight_soc", float64),
    ("sight_wall", float64),
)

spec_agent_neighbour = (
    ("neighbor_radius", float64),
    ("neighborhood_size", int64),
    ("neighbors", int64[:, :]),
    ("neighbor_distances", float64[:, :]),
    ("neighbor_distances_max", float64[:]),
)

spec_agent += spec_agent_motion + spec_agent_neighbour


@numba.jitclass(spec_agent)
class Agent(object):
    """
    Structure for agent parameters and variables.
    """

    def __init__(self, size, mass, radius, ratio_rt, ratio_rs, ratio_ts,
                 inertia_rot, target_velocity, target_angular_velocity):
        """
        Initialise the agent structure.

        Args:
            size (int):
                Number of agents.
            mass (numpy.ndarray):
                Masses of the agents
            radius (numpy.ndarray):
                Total radii of the agents
            ratio_rt (float):
                Ratio of the total radius and torso radius. :math:`[0, 1]`
            ratio_rs (float):
                Ratio of the total radius and shoulder radius. :math:`[0, 1]`
            ratio_ts (float):
                Ratio of the torso radius and torso radius. :math:`[0, 1]`
            inertia_rot (numpy.ndarray):
            target_velocity (numpy.ndarray):
            target_angular_velocity (numpy.ndarray):
        """

        # Array properties
        self.size = size  # Maximum number of agents
        self.shape = (self.size, 2)  # Shape of 2D arrays

        # Agent models (Only one can be active at time).
        self.circular = True
        self.three_circle = False

        # Flags
        self.orientable = self.three_circle  # Orientable has rotational motion
        self.active = np.zeros(size, np.bool8)  # Initialise agents as inactive
        self.goal_reached = np.zeros(size, np.bool8)  # TODO: Deprecate

        # Constant properties
        self.radius = radius  # Total radius
        self.r_t = ratio_rt * radius  # Radius of torso
        self.r_s = ratio_rs * radius  # Radius of shoulders
        self.r_ts = ratio_ts * radius  # Distance from torso to shoulder
        self.mass = mass.reshape(size, 1)  # Mass
        self.inertia_rot = inertia_rot  # Moment of inertia

        # Movement along x and y axis.
        self.position = np.zeros(self.shape)
        self.velocity = np.zeros(self.shape)
        self.target_velocity = target_velocity.reshape(size, 1)
        self.target_direction = np.zeros(self.shape)
        self.force = np.zeros(self.shape)

        # Rotational movement. Three circles agent model
        self.angle = np.zeros(self.size)
        self.angular_velocity = np.zeros(self.size)
        self.target_angle = np.zeros(self.size)
        self.target_angular_velocity = target_angular_velocity
        self.torque = np.zeros(self.size)

        self.position_ls = np.zeros(self.shape)  # Left shoulder
        self.position_rs = np.zeros(self.shape)  # Right shoulder
        self.front = np.zeros(self.shape)  # For plotting agents.
        self.update_shoulder_positions()

        # Motion related parameters
        self.tau_adj = 0.5  # Adjusting force
        self.tau_rot = 0.2  # Adjusting torque
        self.k_soc = 1.5  # Social force scaling
        self.tau_0 = 3.0  # Social force interaction time horizon
        self.mu = 1.2e5  # Contact force
        self.kappa = 4e4  # Contact force
        self.damping = 500  # Contact force
        self.std_rand_force = 0.1  # Fluctuation force
        self.std_rand_torque = 0.1  # Fluctuation torque
        self.f_soc_ij_max = 2e3  # Truncation value for social force
        self.f_soc_iw_max = 2e3  # Truncation value for social force
        self.sight_soc = 3.0  # Interaction distance with other agents
        self.sight_wall = 3.0  # Interaction distance with walls

        # TODO: Move
        # Tracking neighboring agents. Neighbors contains the indices of the
        # neighboring agents. Negative value denotes missing value (if less than
        # neighborhood_size of neighbors).
        self.neighbor_radius = np.nan
        self.neighborhood_size = 8
        self.neighbors = np.ones((self.size, self.neighborhood_size), dtype=np.int64)
        self.neighbor_distances = np.zeros((self.size, self.neighborhood_size))
        self.neighbor_distances_max = np.zeros(self.size)
        self.reset_neighbor()

    def set_circular(self):
        self.circular = True
        self.three_circle = False
        self.orientable = self.three_circle

    def set_three_circle(self):
        self.circular = False
        self.three_circle = True
        self.orientable = self.three_circle

    def reset_motion(self):
        self.force[:] = 0
        self.torque[:] = 0

    def reset_neighbor(self):
        self.neighbors[:, :] = -1  # negative value denotes missing value
        self.neighbor_distances[:, :] = np.inf
        self.neighbor_distances_max[:] = np.inf

    def indices(self):
        """Indices of active agents."""
        # TODO: Other masks
        all_indices = np.arange(self.size)
        return all_indices[self.active]

    def update_shoulder_position(self, i):
        n = np.array((np.cos(self.angle[i]), np.sin(self.angle[i])))
        t = rotate270(n)
        offset = t * self.r_ts[i]
        self.position_ls[i] = self.position[i] - offset
        self.position_rs[i] = self.position[i] + offset
        self.front[i] = self.position[i] + n * self.r_t[i]

    def update_shoulder_positions(self):
        for i in self.indices():
            self.update_shoulder_position(i)
