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
from enum import Enum
from functools import lru_cache

import numba
import numpy as np
from configobj import ConfigObj
from numba import typeof, void, boolean, float64
from numba.types import UniTuple
from sortedcontainers import SortedSet
from validate import Validator

from crowddynamics.core.interactions.distance import distance_circle_circle
from crowddynamics.core.interactions.distance import distance_three_circle
from crowddynamics.core.interactions.partitioning import MutableBlockList
from crowddynamics.core.random.functions import truncnorm
from crowddynamics.core.vector.vector2D import unit_vector, rotate270
from crowddynamics.exceptions import CrowdDynamicsException, OverlappingError, \
    AgentStructureFull, InvalidConfigurationError

BASE_DIR = os.path.dirname(__file__)
AGENT_CFG_SPEC = os.path.join(BASE_DIR, 'agent_spec.cfg')
AGENT_CFG = os.path.join(BASE_DIR, 'agent.cfg')


@lru_cache()
def load_config(infile, configspec):
    """Load configuration from INI file."""
    config = ConfigObj(infile=infile, configspec=configspec)
    if not config.validate(Validator()):
        raise InvalidConfigurationError
    return config


def _truncnorm(mean, abs_scale, size):
    return truncnorm(-3.0, 3.0, loc=mean, abs_scale=abs_scale, size=size)


def body_to_values(body, size):
    """Body to values

    Args:
        body (dict): Dictionary with keys::
        
            ratio_rt = float
            ratio_rs = float
            ratio_ts = float
            radius = float
            radius_scale = float
            velocity = float
            velocity_scale = float
            mass = float
            mass_scale = float
            
        size (int): 
    """
    radius = _truncnorm(body['radius'], body['radius_scale'], size)
    mass = _truncnorm(body['mass'], body['mass_scale'], size)
    # Rotational inertia of mass 80 kg and radius 0.27 m agent.
    # Should be scaled to correct value for agents.
    inertia_rot = 4.0 * (mass / 80.0) * (radius / 0.27) ** 2
    return {
        'r_t': body['ratio_rt'] * radius,
        'r_s': body['ratio_rs'] * radius,
        'r_ts': body['ratio_ts'] * radius,
        'radius': radius,
        'target_velocity': _truncnorm(body['velocity'], body['velocity_scale'],
                                   size),
        'mass': mass,
        'target_angular_velocity': np.full(size, 4 * np.pi),
        'inertia_rot': inertia_rot
    }


def create_random_agent_attributes():
    return {
        'position': np.random.uniform(-1.0, 1.0, 2),
        'target_velocity': np.random.uniform(0.0, 1.0),
        'orientation': np.random.uniform(-np.pi, np.pi),
        'velocity': np.random.uniform(0.0, 1.0, 2),
        'angular_velocity': np.random.uniform(-1.0, 1.0),
        'target_direction': unit_vector(np.random.uniform(-np.pi, np.pi)),
        'target_orientation': np.random.uniform(-np.pi, np.pi)
    }


class AgentModels(Enum):
    """Enumeration class for available agent models."""
    CIRCULAR = 'circular'
    THREE_CIRCLE = 'three_circle'


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

agent_type_circular = np.dtype(translational)
agent_type_three_circle = np.dtype(translational + rotational + three_circle)


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


class AgentManager(object):
    """Class for initialising new agents."""

    def __init__(self, size, model, agent_cfg=AGENT_CFG,
                 agent_cfg_pec=AGENT_CFG_SPEC):
        if model is AgentModels.CIRCULAR:
            self.agents = np.zeros(size, dtype=agent_type_circular)
        elif model is AgentModels.THREE_CIRCLE:
            self.agents = np.zeros(size, dtype=agent_type_three_circle)
        else:
            raise CrowdDynamicsException('Model: {model} in {models}'.format(
                model=model, models=AgentModels
            ))

        self.config = load_config(infile=agent_cfg, configspec=agent_cfg_pec)
        self.constants = self.config['constants']
        self.defaults = self.config['defaults']
        self.body_types = self.config['body_types']

        self.size = size
        self.model = model

        # Keeps track of which agents are active and which in active. Stores
        # indices of agents.
        self.active = SortedSet()
        self.inactive = SortedSet(range(size))

        # Faster check for neighbouring agents for initializing agents into
        # random positions.
        self.grid = MutableBlockList(cell_size=self.constants['cell_size'])

    def add(self, check_overlapping=True, **attributes):
        """Add new agent

        Args:
            check_overlapping (bool): 

            **attributes:
                position (numpy.ndarray): Initial position of the agent
                mass (float): Mass of the agent
                radius (float): Total radius of the agent
                r_t (float): Ratio of the total radius and torso radius. :math:`[0, 1]`
                r_s (float): Ratio of the total radius and shoulder radius. :math:`[0, 1]`
                r_ts (float): Ratio of the torso radius and torso radius. :math:`[0, 1]`
                inertia_rot (float):
                max_velocity (float):
                max_angular_velocity (float):
                orientation (float): Initial orientation :math:`\varphi = [\-pi, \pi]` of the agent.
                velocity (numpy.ndarray): Initial velocity
                angular_velocity (float):
                target_direction (numpy.ndarray): Unit vector to desired direction

        Raises:
            AgentStructureFull: When no more agents can be added.
        """
        try:
            index = self.inactive.pop(0)
        except IndexError:
            raise AgentStructureFull

        # Set default parameters if parameters are not given in attributes
        for key, value in self.defaults.items():
            if key not in attributes:
                attributes[key] = value

        body_type = attributes.get('body_type')
        body = self.body_types[body_type]
        attributes.update(body_to_values(body, 1))

        # Set attributes for agent
        for attribute, value in attributes.items():
            try:
                self.agents[index][attribute] = value
            except KeyError:
                pass  # TODO: warning

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
                raise CrowdDynamicsException

        self.active.add(index)
        return index

    def remove(self, index):
        """Remove agent"""
        if index in self.active:
            self.active.remove(index)
            self.inactive.add(index)
            self.agents[index][:] = 0  # Reset agent
            return True
        else:
            return False

    def fill(self, amount, attributes):
        """Fill agents

        Args:
            amount (int): 
            attributes (dict|Callable[dict]): 
        """
        overlaps = 0
        size = 0
        while size < amount and overlaps < 10 * amount:
            try:
                a = attributes() if callable(attributes) else attributes
                self.add(check_overlapping=True, **a)
                size += 1
            except OverlappingError:
                overlaps += 1
            except AgentStructureFull:
                break

        if self.model is AgentModels.THREE_CIRCLE:
            shoulders(self.agents)
            front(self.agents)
