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
from enum import Enum

import numba
import numpy as np
from numba import typeof, void

from crowddynamics.core.vector.vector2D import unit_vector, rotate270
from crowddynamics.exceptions import CrowdDynamicsException


class AgentModels(Enum):
    """Enumeration class for available agent models."""
    CIRCULAR = 'circular'
    THREE_CIRCLE = 'three_circle'


class AgentBodyTypes(Enum):
    ADULT = 'adult'
    MALE = 'male'
    FEMALE = 'female'
    CHILD = 'child'
    ELDERY = 'eldery'


# Default values
tau_adj = 0.5
tau_rot = 0.2
k_soc = 1.5
tau_0 = 3.0
mu = 1.2e5
kappa = 4e4
damping = 500
std_rand_force = 0.1
std_rand_torque = 0.1

# Limits
sight_soc = 3.0
sight_wall = 3.0
f_soc_ij_max = 2e3
f_soc_iw_max = 2e3

# Agent attributes
translational = [
    ('mass', np.float64),
    ('radius', np.float64),
    ('position', np.float64, 2),
    ('velocity', np.float64, 2),
    ('target_velocity', np.float64),
    ('target_direction', np.float64, 2),
    ('force', np.float64, 2),
    ('tau_adj', np.float64),
    ('k_soc', np.float64),
    ('tau_0', np.float64),
    ('mu', np.float64),
    ('kappa', np.float64),
    ('damping', np.float64),
    ('std_rand_force', np.float64),
    ('f_soc_ij_max', np.float64),
    ('f_soc_iw_max', np.float64),
    ('sight_soc', np.float64),
    ('sight_wall', np.float64),
]

rotational = [
    ('inertia_rot', np.float64),
    ('orientation', np.float64),
    ('angular_velocity', np.float64),
    ('target_orientation', np.float64),
    ('target_angular_velocity', np.float64),
    ('torque', np.float64),
    ('tau_rot', np.float64),
    ('std_rand_torque', np.float64),
]

three_circle = [
    ('r_t', np.float64),
    ('r_s', np.float64),
    ('r_ts', np.float64),
    ('position_ls', np.float64, 2),
    ('position_rs', np.float64, 2),
    ('front', np.float64, 2),
]

# Agent model types
agent_type_circular = np.dtype(
    translational
)

agent_type_three_circle = np.dtype(
    translational +
    rotational +
    three_circle
)


@numba.jit(void(typeof(agent_type_three_circle)[:]),
           nopython=True, nogil=True, cache=True)
def shoulders(agents):
    """Positions of the center of mass, left- and right shoulders.

    Args:
        agents (ndarray):
            Numpy array of datatype ``dtype=agent_type_three_circle``.
    """
    for agent in agents:
        tangent = rotate270(unit_vector(agent.orientation))
        offset = tangent * agent.r_ts
        agent.position_ls[:] = agent.position - offset
        agent.position_rs[:] = agent.position + offset


@numba.jit(void(typeof(agent_type_three_circle)[:]),
           nopython=True, nogil=True, cache=True)
def front(agents):
    """Position of agents nose."""
    for agent in agents:
        agent.front[:] = agent.position * unit_vector(agent.orientation) * \
                         agent.r_t


@numba.generated_jit(cache=True)
def reset_motion(agent):
    agent.force[:] = 0
    if agent.dtype is numba.from_dtype(agent_type_three_circle):
        agent.torque[:] = 0


class AgentManager(object):
    def __init__(self, size, model):
        if model is AgentModels.CIRCULAR:
            self.agents = np.zeros(size, dtype=agent_type_circular)
        elif model is AgentModels.THREE_CIRCLE:
            self.agents = np.zeros(size, dtype=agent_type_three_circle)
        else:
            CrowdDynamicsException('Model: {model} in in {models}'.format(
                model=model, models=AgentModels
            ))
        self.size = size
        self.model = model
        self.active = np.zeros(self.size, dtype=np.bool8)

    def add(self, index, **attributes):
        agent = self.agents[index]
        self.set_attributes(index, **attributes)
        self.active[index] = np.bool8(True)

    def remove(self, index):
        agent = self.agents[index]
        self.active[index] = np.bool8(False)

    def set_attributes(self, index, **attributes):
        agent = self.agents[index]
        for attribute, value in attributes.items():
            agent[attribute] = value


# Linear obstacle defined by two points
obstacle_type_linear = np.dtype([
    ('p0', np.float64, 2),
    ('p1', np.float64, 2),
])


# Neighborhood for tracking neighboring agents
Neighborhood = namedtuple(
    'Neighborhood', ['neighbor_radius', 'neighborhood_size', 'neighbors']
)


def init_neighborhood(agent_size, neighborhood_size, neighbor_radius):
    """Initialise neighborhood

    Args:
        agent_size (int):
        neighborhood_size (int):
        neighbor_radius (float):

    Returns:
        Neighborhood:
    """
    dtype = np.dtype([
        ('agent_indices', np.int64, neighborhood_size),
        ('distances', np.float64, neighborhood_size),
        ('distances_max', np.float64),
    ])
    neighbors = np.zeros(agent_size, dtype=dtype)
    neighborhood = Neighborhood(neighbor_radius, neighborhood_size, neighbors)
    reset_neighborhood(neighborhood)
    return neighborhood


def reset_neighborhood(neighborhood):
    neighborhood.neighbors['agent_indices'] = -1
    neighborhood.neighbors['distances'] = np.inf
    neighborhood.neighbors['distances_max'] = np.inf
