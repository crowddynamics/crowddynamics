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
import os
from collections import namedtuple
from enum import Enum

import numba
import numpy as np
import pandas as pd
from numba import typeof, void, boolean, float64
from numba.types import UniTuple
from sortedcontainers import SortedSet

from crowddynamics.core.interactions import distance_circle_circle
from crowddynamics.core.interactions import distance_three_circle
from crowddynamics.core.interactions.partitioning import MutableBlockList
from crowddynamics.core.vector.vector2D import unit_vector, rotate270
from crowddynamics.exceptions import CrowdDynamicsException, OverlappingError, \
    AgentStructureFull


# TODO: read configutaion from .cfg file
BASE_DIR = os.path.dirname(__file__)
BODIES = pd.read_csv(os.path.join(BASE_DIR, 'body.csv'), index_col=[0])


class AgentModels(Enum):
    """Enumeration class for available agent models."""
    CIRCULAR = 'circular'
    THREE_CIRCLE = 'three_circle'


class AgentBodyTypes(Enum):
    """Enumeration for available agent body types."""
    ADULT = 'adult'
    MALE = 'male'
    FEMALE = 'female'
    CHILD = 'child'
    ELDERY = 'eldery'


class Defaults(object):
    """Default values for agent parameters"""
    orientation = 0.0

    # Motion parameters
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

    @classmethod
    def items(cls):
        """Yield items from a class."""
        for key, value in cls.__dict__.items():
            if not key.startswith('__'):
                yield key, value

    @classmethod
    def to_dict(cls):
        return {key: value for key, value in cls.items()}


MAX_AGENT_RADIUS = 0.3


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
    def _reset(agent):
        agent['force'][:] = 0

    def _reset2(agent):
        agent['force'][:] = 0
        agent['torque'][:] = 0

    if agent.dtype is numba.from_dtype(agent_type_circular):
        return _reset
    if agent.dtype is numba.from_dtype(agent_type_three_circle):
        return _reset2


@numba.jit([boolean(typeof(agent_type_circular)[:], float64[:], float64)],
           nopython=True, nogil=True, cache=True)
def overlapping_circle_circle(agents, x, r):
    """Test if two circles are overlapping.

    Args:
        agents:
        x: Position of agent that is tested
        r: Radius of agent that is tested

    Returns:
        bool:
    """
    for agent in agents:
        h, _ = distance_circle_circle(agent.position, agent.radius, x, r)
        if h < 0.0:
            return True
    return False


@numba.jit([boolean(typeof(agent_type_three_circle)[:],
                    UniTuple(float64[:], 3), UniTuple(float64, 3))],
           nopython=True, nogil=True, cache=True)
def overlapping_three_circle(agents, x, r):
    """Test if two three-circle models are overlapping.

    Args:
        x1: Positions of other agents
        r1: Radii of other agents
        x: Position of agent that is tested
        r: Radius of agent that is tested

    Returns:
        bool:

    """
    for agent in agents:
        h, _, _, _ = distance_three_circle(
            (agent.position, agent.position_ls, agent.position_rs),
            (agent.r_t, agent.r_s, agent.r_s),
            x, r
        )
        if h < 0:
            return True
    return False


def create_random_agent_attributes():
    return dict(
        position=np.random.uniform(-1.0, 1.0, 2),
        mass=np.random.uniform(0.0, 1.0),
        radius=np.random.uniform(0.0, 1.0),
        r_t=np.random.uniform(0.0, 1.0),
        r_s=np.random.uniform(0.0, 1.0),
        r_ts=np.random.uniform(0.0, 1.0),
        inertia_rot=np.random.uniform(0.0, 1.0),
        target_velocity=np.random.uniform(0.0, 1.0),
        target_angular_velocity=np.random.uniform(0.0, 1.0),
        orientation=np.random.uniform(-np.pi, np.pi),
        velocity=np.random.uniform(0.0, 1.0, 2),
        angular_velocity=np.random.uniform(-1.0, 1.0),
        target_direction=unit_vector(np.random.uniform(-np.pi, np.pi)),
        target_orientation=np.random.uniform(-np.pi, np.pi),
    )


class AgentManager(object):
    """Class for initialising new agents."""

    def __init__(self, size, model):
        if model is AgentModels.CIRCULAR:
            self.agents = np.zeros(size, dtype=agent_type_circular)
        elif model is AgentModels.THREE_CIRCLE:
            self.agents = np.zeros(size, dtype=agent_type_three_circle)
        else:
            raise CrowdDynamicsException('Model: {model} in in {models}'.format(
                model=model, models=AgentModels
            ))

        self.size = size
        self.model = model

        # Keeps track of which agents are active and which in active. Stores
        # indices of agents.
        self.active = SortedSet()
        self.inactive = SortedSet(range(size))

        # Faster check for neighbouring agents for initializing agents into
        # random positions.
        self.grid = MutableBlockList(cell_size=MAX_AGENT_RADIUS)

    def add(self, check_overlapping=True, **attributes):
        """Add new agent

        Args:
            index (int):

            **attributes:
                position (numpy.ndarray):
                    Initial position of the agent

                mass (float):
                    Mass of the agent

                radius (float):
                    Total radius of the agent

                r_t (float):
                    Ratio of the total radius and torso radius. :math:`[0, 1]`

                r_s (float):
                    Ratio of the total radius and shoulder radius. :math:`[0, 1]`

                r_ts (float):
                    Ratio of the torso radius and torso radius. :math:`[0, 1]`

                inertia_rot (float):

                max_velocity (float):

                max_angular_velocity (float):

                orientation (float):
                    Initial orientation :math:`\varphi = [\-pi, \pi]` of the agent.

                velocity (numpy.ndarray):
                    Initial velocity

                angular_velocity (float):

                target_direction (numpy.ndarray):
                    Unit vector to desired direction

        Raises:

        Returns:

        """
        if self.inactive:
            index = self.inactive.pop(0)

            # TODO: default parameters
            # Set default parameters if parameters are not given in attributes
            for key, value in Defaults.items():
                if key not in attributes:
                    attributes[key] = value

            self.set_attributes(index, **attributes)

            # Update shoulder positions for three circle agents
            if self.model is AgentModels.THREE_CIRCLE:
                shoulders(self.agents)

            # Check if agents are overlapping.
            position = attributes.get('position')
            orientation = attributes.get('orientation')
            radius = attributes.get('radius')

            if check_overlapping and position:
                neighbours = self.grid[position]
                agents = self.agents[neighbours]

                if self.model is AgentModels.CIRCULAR:
                    if overlapping_circle_circle(agents, position, radius):
                        self.agents[index][:] = 0
                        raise OverlappingError()
                elif self.model is AgentModels.THREE_CIRCLE:
                    if overlapping_three_circle(agents, None, None):
                        self.agents[index][:] = 0
                        raise OverlappingError()
                else:
                    raise CrowdDynamicsException("")

            self.active.add(index)
            return index
        else:
            return AgentStructureFull()

    def remove(self, index):
        """Remove agent"""
        if index in self.active:
            self.active.remove(index)
            self.inactive.add(index)
            self.agents[index][:] = 0  # Reset agent
            return True
        else:
            return False

    def set_attributes(self, index, **attributes):
        """Set attribute value for agent"""
        agent = self.agents[index]
        for attribute, value in attributes.items():
            if attribute in agent.dtype.names:
                agent[attribute] = value

    def fill_random(self, seed=None):
        """Fill with random agents for testing purposes"""
        np.random.seed(seed)
        while self.inactive:
            self.add(**create_random_agent_attributes())
        if self.model is AgentModels.THREE_CIRCLE:
            shoulders(self.agents)
            front(self.agents)


# Linear obstacle defined by two points
obstacle_type_linear = np.dtype([
    ('p0', np.float64, 2),
    ('p1', np.float64, 2),
])


class ObstacleManager(object):
    pass


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
    missing = -1
    neighborhood.neighbors['agent_indices'] = missing
    neighborhood.neighbors['distances'] = np.inf
    neighborhood.neighbors['distances_max'] = np.inf
