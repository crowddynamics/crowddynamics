r"""Agent model for multiagent simulation

**Circular model**

Simplest of the models is circular model without orientation. Circle is defined
with radius :math:`r > 0` from the center of mass.

**Three circle model** :cite:`Langston2006`

Three circle model models agent with three circles which represent torso and two
shoulders. Torso has radius of :math:`r_t` and is centered at center of mass
:math:`\mathbf{x}` and shoulder have both radius of  :math:`r_s` and are
centered at :math:`\mathbf{x} \pm r_{ts} \mathbf{\hat{e}_t}` where normal and
tangent vectors

.. math::
   \mathbf{\hat{e}_n} &= [\cos(\varphi), \sin(\varphi)] \\
   \mathbf{\hat{e}_t} &= [\sin(\varphi), -\cos(\varphi)]

"""
import numba
import numpy as np
from numba import float64, int64, boolean, f8
from numba.types import UniTuple

from crowddynamics.core.vector2D.vector2D import rotate270, wrap_to_pi


@numba.jit(UniTuple(f8[:], 3)(f8[:], f8, f8), nopython=True, nogil=True)
def positions_scalar(position, orientation, radius_ts):
    """Center and shoulder positions"""
    x = np.cos(orientation)
    y = np.sin(orientation)
    n = np.array((x, y))
    t = rotate270(n)
    offset = t * radius_ts
    position_ls = position - offset
    position_rs = position + offset
    return position, position_ls, position_rs


@numba.jit(UniTuple(f8[:, :], 3)(f8[:, :], f8[:], f8[:]), nopython=True, nogil=True)
def positions_vector(position, orientation, radius_ts):
    """Center and shoulder positions"""
    x = np.cos(orientation)
    y = np.sin(orientation)
    t = np.stack((y, -x), axis=1)
    offset = t * radius_ts
    position_ls = position - offset
    position_rs = position + offset
    return position, position_ls, position_rs


@numba.generated_jit(nopython=True, nogil=True)
def positions(position, orientation, radius_ts):
    if isinstance(orientation, numba.types.Float):
        return lambda position, orientation, radius_ts: \
            positions_scalar(position, orientation, radius_ts)
    elif isinstance(orientation, numba.types.Array):
        return lambda position, orientation, radius_ts: \
            positions_vector(position, orientation, radius_ts)
    else:
        raise Exception()


def resize():
    return NotImplementedError


spec_agent = (
    ("size", int64),
    ("shape", UniTuple(int64, 2)),
    ("circular", boolean),
    ("three_circle", boolean),
    ("orientable", boolean),
    ("active", boolean[:]),
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
    ("orientation", float64[:]),
    ("angular_velocity", float64[:]),
    ("target_orientation", float64[:]),
    ("target_angular_velocity", float64[:]),
    ("torque", float64[:]),
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
    # ("neighbor_radius", float64),
    # ("neighborhood_size", int64),
    # ("neighbors", int64[:, :]),
    # ("neighbor_distances", float64[:, :]),
    # ("neighbor_distances_max", float64[:]),
)

spec_agent += spec_agent_motion  # + spec_agent_neighbour


@numba.jitclass(spec_agent)
class Agent(object):
    r"""Structure for agent parameters and variables.

    Args:
        size (int):
            Maximum number of agents :math:`N`.
        shape (tuple):
            Shape of 2D arrays :math:`(N, 2)`.
        circular (bool):
            Boolean indicating if agent is modeled as a circle
        three_circle (bool):
            Boolean indicating if agent is modeled as three circles
        orientable (bool):
            Boolean indicating if agent is orientable (has rotational motion).
        active:
        radius:
            Radius :math:`r > 0`
        r_t:
            Radius of torso :math:`r_t > 0`
        r_s:
            Radius of shoulder :math:`t_s > 0`
        r_ts:
            Distance from torso to shoulder :math:`r_{ts}`
        mass:
            Mass :math:`m > 0`
        inertia_rot:
            Moment of inertia :math:`I_{rot} > 0`
        position:
            Center of the mass :math:`\mathbf{x}`
        velocity:
            Velocity :math:`\mathbf{v}`
        target_velocity:
            Target velocity :math:`v_{0}`
        target_direction:
            Target direction :math:`\mathbf{e}_0`
        force:
            Force :math:`\mathbf{f}`
        orientation:
            Orientation :math:`\varphi \in [-\pi, \pi]`
        angular_velocity:
            Angular velocity :math:`\omega`
        target_orientation:
            Target orientation :math:`\varphi_0 \in [-\pi, \pi]`
        target_angular_velocity:
            Target Angular velocity :math:`\omega_0`
        torque:
            Torque :math:`M`
        tau_adj:
            Characteristic time for agent adjusting its movement
        tau_rot:
            Characteristic time for agent adjusting its rotational movement
        k_soc:
            Social force scaling constant
        tau_0:
            Interaction time horizon
        mu:
            Compression counteraction constant
        kappa:
            Sliding friction constant
        damping:
            Damping coefficient for contact force
        std_rand_force:
            Standard deviation for random force from truncated normal
            distribution
        std_rand_torque:
            Standard deviation for random torque from truncated normal
            distribution
        f_soc_ij_max:
            Truncation for social force with agent to agent interaction
        f_soc_iw_max:
            Truncation for social force with agent to wall interaction
        sight_soc:
            Maximum distance for social force to effect
        sight_wall:
            Maximum distance for social force to effect

    """

    def __init__(self, size):
        r"""Initialise the agent structure.

        Args:
            size (int):
                Maximum number of agents that can be placed into the structure.

        """
        self.size = size
        self.shape = (self.size, 2)

        # Flags
        self.circular = True
        self.three_circle = False
        self.orientable = False
        self.active = np.zeros(size, np.bool8)  # Initialise agents as inactive

        # Constant properties
        self.radius = np.zeros(self.size)       # Total radius
        self.r_t = np.zeros(self.size)          # Radius of torso
        self.r_s = np.zeros(self.size)          # Radius of shoulders
        self.r_ts = np.zeros(self.size)         # Distance from torso to shoulder
        self.mass = np.zeros((self.size, 1))    # Mass
        self.inertia_rot = np.zeros(self.size)  # Moment of inertia

        # Movement along x and y axis.
        self.position = np.zeros(self.shape)
        self.velocity = np.zeros(self.shape)
        self.target_velocity = np.zeros((size, 1))
        self.target_direction = np.zeros(self.shape)
        self.force = np.zeros(self.shape)

        # Rotational movement. Three circles agent model
        self.orientation = np.zeros(self.size)
        self.angular_velocity = np.zeros(self.size)
        self.target_orientation = np.zeros(self.size)
        self.target_angular_velocity = np.zeros(self.size)
        self.torque = np.zeros(self.size)

        # Motion related parameters
        # TODO: arrays
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

        # Tracking neighboring agents. Neighbors contains the indices of the
        # neighboring agents. Negative value denotes missing value (if less than
        # neighborhood_size of neighbors).
        # TODO: move to interactions
        # self.neighbor_radius = np.nan
        # self.neighborhood_size = 8
        # self.neighbors = np.ones((self.size, self.neighborhood_size), dtype=np.int64)
        # self.neighbor_distances = np.zeros((self.size, self.neighborhood_size))
        # self.neighbor_distances_max = np.zeros(self.size)
        # self.reset_neighbor()

    def add(self, position, mass, radius, ratio_rt, ratio_rs, ratio_ts,
            inertia_rot, max_velocity, max_angular_velocity):
        r"""
        Add new agent to next free index if there is space left.

        Args:
            position (numpy.ndarray):
                Initial position of the agent

            mass (float):
                Mass of the agent

            radius (float):
                Total radius of the agent

            ratio_rt (float):
                Ratio of the total radius and torso radius. :math:`[0, 1]`

            ratio_rs (float):
                Ratio of the total radius and shoulder radius. :math:`[0, 1]`

            ratio_ts (float):
                Ratio of the torso radius and torso radius. :math:`[0, 1]`

            inertia_rot (float):

            max_velocity (float):

            max_angular_velocity (float):

        Returns:
            int: Integer indicating the index of agent that was added.
                 Returns -1 if addition was unsuccessful.

        """
        # mass, radius, ratio_rt, ratio_rs, ratio_ts,
        # inertia_rot, target_velocity, target_angular_velocity

        # Find first inactive agent
        for i, state in enumerate(self.active):
            if state:
                continue
            else:
                self.active[i] = True
                self.position[i] = position
                self.mass[i] = mass
                self.radius[i] = radius
                self.r_t[i] = ratio_rt * radius
                self.r_s[i] = ratio_rs * radius
                self.r_ts[i] = ratio_ts * radius
                self.inertia_rot[i] = inertia_rot
                self.target_velocity[i] = max_velocity
                self.target_angular_velocity[i] = max_angular_velocity
                return i
        return -1

    def remove(self, i):
        r"""
        Remove agent of ``index``.
        - Set agent inactive

        Args:
            i (int):
        """
        self.active[i] = False

    def set_circular(self):
        self.circular = True
        self.three_circle = False
        self.orientable = self.three_circle

    def set_three_circle(self):
        self.circular = False
        self.three_circle = True
        self.orientable = self.three_circle

    def set_motion(self, i, orientation, velocity, angular_velocity,
                   target_direction, target_orientation):
        r"""
        Set motion parameters for agent.

        Args:

            i:

            orientation (float):
                Initial orientation :math:`\varphi = [\-pi, \pi]` of the agent.

            velocity (numpy.ndarray):
                Initial velocity

            angular_velocity (float):

            target_direction (numpy.ndarray):

        Returns:
            bool:

        """
        if self.active[i]:
            self.orientation[i] = wrap_to_pi(orientation)
            self.velocity[i] = velocity
            self.angular_velocity[i] = angular_velocity
            self.target_direction[i] = target_direction
            self.target_orientation[i] = target_orientation
            return True
        else:
            return False

    def positions(self, i):
        r"""
        Positions of the center of mass, left- and right shoulders

        Args:
            i (int):

        Returns:
            (numpy.ndarray, numpy.ndarray, numpy.ndarray):
                - Center of mass
                - Left shoulder
                - Right shoulder
        """
        return positions(self.position[i], self.orientation[i], self.r_ts[i])

    def radii(self, i):
        r"""
        Radii of torso and shoulders.

        Args:
            i:

        Returns:
            (float, float, float):
                - Radius torso
                - Radius left shoulder
                - Radius right shoulder
        """
        return self.r_t[i], self.r_s[i], self.r_s[i]

    def front(self, i):
        n = np.array((np.cos(self.orientation[i]), np.sin(self.orientation[i])))
        position = self.position[i]
        return position + n * self.r_t[i]

    def reset_motion(self):
        self.force[:] = 0
        self.torque[:] = 0

    # def reset_neighbor(self):
    #     self.neighbors[:, :] = -1  # negative value denotes missing value
    #     self.neighbor_distances[:, :] = np.inf
    #     self.neighbor_distances_max[:] = np.inf

    def indices(self):
        """Indices of active agents."""
        return np.arange(self.size)[self.active]
