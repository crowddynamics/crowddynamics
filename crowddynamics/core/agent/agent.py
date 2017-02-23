r"""Agent model for multiagent simulation

**Circular model**

Simplest of the models is circular model without orientation. Circle is defined
with radius :math:`r > 0` from the center of mass.

**Three circle model** [Langston2006]_

Three circle model models agent with three circles which represent torso and two
shoulders. Torso has radius of :math:`r_t` and is centered at center of mass
:math:`\mathbf{x}` and shoulder have both radius of  :math:`r_s` and are
centered at :math:`\mathbf{x} \pm r_{ts} \mathbf{\hat{e}_t}` where normal and
tangent vectors

.. math::
   \mathbf{\hat{e}_n} &= [\cos(\varphi), \sin(\varphi)] \\
   \mathbf{\hat{e}_t} &= [\sin(\varphi), -\cos(\varphi)]

"""
from collections import namedtuple

import numba
import numpy as np
from numba import float64, int64, boolean, f8
from numba.types import UniTuple

from crowddynamics.core.vector.vector2D import rotate270, wrap_to_pi


Attribute = namedtuple('Attribute', ('name', 'numba_type', 'resizable'))
AGENT_ATTRS = (
    Attribute('size', int64, False),
    Attribute('shape', UniTuple(int64, 2), False),
    Attribute('circular', boolean, False),
    Attribute('three_circle', boolean, False),
    Attribute('orientable', boolean, False),
    Attribute('active', boolean[:], True),
    Attribute('mass', float64[:, :], False),
    Attribute('radius', float64[:], False),
    Attribute('r_t', float64[:], False),
    Attribute('r_s', float64[:], False),
    Attribute('r_ts', float64[:], False),
    Attribute('position', float64[:, :], True),
    Attribute('velocity', float64[:, :], True),
    Attribute('target_velocity', float64[:, :], True),
    Attribute('target_direction', float64[:, :], True),
    Attribute('force', float64[:, :], True),
    Attribute('inertia_rot', float64[:], True),
    Attribute('orientation', float64[:], True),
    Attribute('angular_velocity', float64[:], True),
    Attribute('target_orientation', float64[:], True),
    Attribute('target_angular_velocity', float64[:], True),
    Attribute('torque', float64[:], True),
    Attribute('tau_adj', float64[:, :], False),
    Attribute('tau_rot', float64[:, :], False),
    Attribute('k_soc', float64[:], False),
    Attribute('tau_0', float64[:], False),
    Attribute('mu', float64[:], False),
    Attribute('kappa', float64[:], False),
    Attribute('damping', float64[:], False),
    Attribute('std_rand_force', float64[:], False),
    Attribute('std_rand_torque', float64[:], False),
    Attribute('f_soc_ij_max', float64, False),
    Attribute('f_soc_iw_max', float64, False),
    Attribute('sight_soc', float64, False),
    Attribute('sight_wall', float64, False),
)


NEIGHBORHOOD_SPEC = (
    ('neighbor_radius', float64),
    ('neighborhood_size', int64),
    ('neighbors', int64[:, :]),
    ('neighbor_distances', float64[:, :]),
    ('neighbor_distances_max', float64[:]),
)


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


@numba.jitclass(tuple((p.name, p.numba_type) for p in AGENT_ATTRS))
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
        self.active = np.zeros(self.size, np.bool8)

        # Agent properties
        self.radius = np.zeros(self.size)
        self.r_t = np.zeros(self.size)
        self.r_s = np.zeros(self.size)
        self.r_ts = np.zeros(self.size)
        self.mass = np.zeros((self.size, 1))
        self.inertia_rot = np.zeros(self.size)

        # Translational motion
        self.position = np.zeros(self.shape)
        self.velocity = np.zeros(self.shape)
        self.target_velocity = np.zeros((size, 1))
        self.target_direction = np.zeros(self.shape)
        self.force = np.zeros(self.shape)

        # Rotational motion
        self.orientation = np.zeros(self.size)
        self.angular_velocity = np.zeros(self.size)
        self.target_orientation = np.zeros(self.size)
        self.target_angular_velocity = np.zeros(self.size)
        self.torque = np.zeros(self.size)

        # Motion related parameters
        self.tau_adj = 0.5 * np.ones((self.size, 1))
        self.tau_rot = 0.2 * np.ones((self.size, 1))
        self.k_soc = 1.5 * np.ones(self.size)
        self.tau_0 = 3.0 * np.ones(self.size)
        self.mu = 1.2e5 * np.ones(self.size)
        self.kappa = 4e4 * np.ones(self.size)
        self.damping = 500 * np.ones(self.size)
        self.std_rand_force = 0.1 * np.ones(self.size)
        self.std_rand_torque = 0.1 * np.ones(self.size)

        # Limits
        self.sight_soc = 3.0
        self.sight_wall = 3.0
        self.f_soc_ij_max = 2e3
        self.f_soc_iw_max = 2e3

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
                Unit vector to desired direction

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

    def indices(self):
        """Indices of active agents."""
        return np.arange(self.size)[self.active]


Agent_numba_type = Agent.class_type.instance_type


class NeighborHood(object):
    # Tracking neighboring agents. Neighbors contains the indices of the
    # neighboring agents. Negative value denotes missing value (if less than
    # neighborhood_size of neighbors).
    # TODO: move to interactions

    def __init__(self, size):
        self.size = size
        self.neighbor_radius = np.nan
        self.neighborhood_size = 8
        self.neighbors = np.ones((self.size, self.neighborhood_size), dtype=np.int64)
        self.neighbor_distances = np.zeros((self.size, self.neighborhood_size))
        self.neighbor_distances_max = np.zeros(self.size)
        self.reset_neighbor()

    def reset_neighbor(self):
        self.neighbors[:, :] = -1  # negative value denotes missing value
        self.neighbor_distances[:, :] = np.inf
        self.neighbor_distances_max[:] = np.inf
