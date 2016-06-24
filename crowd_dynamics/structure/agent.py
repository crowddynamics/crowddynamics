import numba
import numpy as np
from numba import float64, int64, boolean
from numba.types import UniTuple

spec_agent = (
    ("size", int64),
    ("shape", UniTuple(int64, 2)),
    ("three_circles_flag", boolean),
    ("orientable_flag", boolean),
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
    ("force_adjust", float64[:, :]),
    ("force_agent", float64[:, :]),
    ("force_wall", float64[:, :]),
    ("inertia_rot", float64[:]),
    ("angle", float64[:]),
    ("angular_velocity", float64[:]),
    ("target_angle", float64[:]),
    ("target_angular_velocity", float64[:]),
    ("torque", float64[:]),
    ("position_ls", float64[:, :]),
    ("position_rs", float64[:, :]),
    ("front", float64[:, :]),
    ("tau_adj", float64),
    ("tau_adj_rot", float64),
    ("k", float64),
    ("tau_0", float64),
    ("mu", float64),
    ("kappa", float64),
    ("damping", float64),
    ("a", float64),
    ("b", float64),
    ("std_rand_force", float64),
    ("std_rand_torque", float64),
    ("f_soc_ij_max", float64),
    ("f_soc_iw_max", float64),
    ("sight_soc", float64),
    ("sight_wall", float64),
)

agent_attr_names = [item[0] for item in spec_agent]


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
                 target_velocity,
                 target_angular_velocity,
                 three_circles_flag):
        # Array properties
        self.size = size
        self.shape = (size, 2)

        # Flags
        self.three_circles_flag = three_circles_flag
        self.orientable_flag = self.three_circles_flag
        self.active = np.ones(size, np.bool8)
        self.goal_reached = np.zeros(size, np.bool8)

        # Constant properties
        self.radius = radius  # Total radius
        self.r_t = radius_torso  # Radius of torso
        self.r_s = radius_shoulder  # Radius of shoulders
        self.r_ts = radius_torso_shoulder  # Distance from torso to shoulder
        self.mass = mass.reshape(size, 1)  # Mass
        self.inertia_rot = inertia_rot  # Moment of inertia

        # Movement along x and y axis. Circular agent model
        self.position = np.zeros(self.shape)
        self.velocity = np.zeros(self.shape)
        self.target_velocity = target_velocity.reshape(size, 1)
        self.target_direction = np.zeros(self.shape)
        self.force = np.zeros(self.shape)
        self.force_adjust = np.zeros(self.shape)
        self.force_agent = np.zeros(self.shape)
        self.force_wall = np.zeros(self.shape)

        # Rotational movement. Three circles agent model
        self.angle = np.zeros(self.size)
        self.angular_velocity = np.zeros(self.size)
        self.target_angle = np.zeros(self.size)
        self.target_angular_velocity = target_angular_velocity
        self.torque = np.zeros(self.size)

        self.position_ls = np.zeros(self.shape)  # Left shoulder
        self.position_rs = np.zeros(self.shape)  # Right shoulder
        self.front = np.zeros(self.shape)  # For plotting agents.

        # TODO: vector form, table of values
        # Force related parameters
        self.tau_adj = 0.5
        self.tau_adj_rot = 0.2
        self.k = 1.5 * 70  # 1.5 * mass?
        self.tau_0 = 3.0
        self.mu = 1.2e5
        self.kappa = 4e4
        self.damping = 500
        self.a = 2000
        self.b = 0.08

        # Standard deviation for truncated normal distribution
        self.std_rand_force = 0.1
        self.std_rand_torque = 0.1

        # Cutoff value for social force
        self.f_soc_ij_max = 2e3
        self.f_soc_iw_max = 2e3

        # Cutoff distance
        self.sight_soc = 7.0
        self.sight_wall = 7.0

    def reset(self):
        self.force *= 0
        self.torque *= 0
        self.force_adjust *= 0
        self.force_agent *= 0
        self.force_wall *= 0

    def update_shoulder_positions(self):
        for i in range(self.size):
            n = np.array((np.cos(self.angle[i]), np.sin(self.angle[i])))
            t = np.array((-np.sin(self.angle[i]), np.cos(self.angle[i])))
            offset = t * self.r_ts[i]
            self.position_ls[i] = self.position[i] - offset
            self.position_rs[i] = self.position[i] + offset
            self.front[i] = self.position[i] + n * self.r_t[i]
