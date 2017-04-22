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


Attributes:
    BASE_DIR:
    AGENT_CFG_SPEC:
    AGENT_CFG:
    translational:
    rotational:
    three_circle:
    agent_type_circular:
    agent_type_three_circle:
    
"""
import logging

import numba
import numpy as np
from loggingtools import log_with
from numba import typeof, void, boolean, float64
from numba.types import UniTuple
from sortedcontainers import SortedSet

from crowddynamics.config import load_config, AGENT_CFG, AGENT_CFG_SPEC
from crowddynamics.core.interactions.distance import distance_circles, \
    distance_circle_line, distance_three_circle_line
from crowddynamics.core.interactions.distance import distance_three_circles
from crowddynamics.core.interactions.partitioning import MutableBlockList
from crowddynamics.core.random.functions import truncnorm
from crowddynamics.core.structures.obstacles import obstacle_type_linear
from crowddynamics.core.vector.vector2D import unit_vector, rotate270
from crowddynamics.exceptions import CrowdDynamicsException, OverlappingError, \
    AgentStructureFull
from tqdm import tqdm


def _truncnorm(mean, abs_scale):
    """Individual value from truncnorm"""
    return np.asscalar(
        truncnorm(-3.0, 3.0, loc=mean, abs_scale=abs_scale, size=1))


def body_to_values(body):
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
    """
    radius = _truncnorm(body['radius'], body['radius_scale'])
    mass = _truncnorm(body['mass'], body['mass_scale'])
    # Rotational inertia of mass 80 kg and radius 0.27 m agent.
    # Should be scaled to correct value for agents.
    inertia_rot = 4.0 * (mass / 80.0) * (radius / 0.27) ** 2
    return {
        'r_t': body['ratio_rt'] * radius,
        'r_s': body['ratio_rs'] * radius,
        'r_ts': body['ratio_ts'] * radius,
        'radius': radius,
        'target_velocity': _truncnorm(body['velocity'], body['velocity_scale']),
        'mass': mass,
        'target_angular_velocity': 4 * np.pi,
        'inertia_rot': inertia_rot
    }


translational = [
    ('mass', np.float64),
    ('radius', np.float64),
    ('position', np.float64, 2),
    ('velocity', np.float64, 2),
    ('target_velocity', np.float64),
    ('target_direction', np.float64, 2),
    ('force', np.float64, 2),
    ('force_prev', np.float64, 2),
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
    ('torque_prev', np.float64),
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


AgentModelToType = {
    'circular': agent_type_circular,
    'three_circle': agent_type_three_circle,
}

AgentTypeToModel = {
    agent_type_circular: 'circular',
    agent_type_three_circle: 'three_circle',
}

AgentModels = list(AgentModelToType)


def register_agent_model(name, dtype):
    AgentModelToType[name] = dtype
    AgentTypeToModel[dtype] = name


def is_model(agents, model):
    """Test if agent if type same type as model name
    
    Args:
        agents (numpy.ndarray): 
        model (str): 

    Returns:
        bool:
    """
    return hash(agents.dtype) == hash(AgentModelToType[model])


def of_model(agents):
    """Returns the model name of agents"""
    return AgentTypeToModel[agents.dtype]


@numba.jit(void(typeof(agent_type_three_circle)[:]),
           nopython=True, nogil=True, cache=True)
def shoulders(agents):
    """Positions of the center of mass, left- and right shoulders.

    Args:
        agents (ndarray):
            Numpy array of datatype ``dtype=agent_type_three_circle``.
    """
    for agent in agents:
        tangent = rotate270(unit_vector(agent['orientation']))
        offset = tangent * agent['r_ts']
        agent['position_ls'][:] = agent['position'] - offset
        agent['position_rs'][:] = agent['position'] + offset


@numba.jit(void(typeof(agent_type_three_circle)[:]),
           nopython=True, nogil=True, cache=True)
def front(agents):
    """Position of agents nose."""
    for agent in agents:
        agent['front'][:] = agent['position'] * \
                            unit_vector(agent['orientation']) * \
                            agent['r_t']


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
def overlapping_circles(agents, x, r):
    """Test if two circles are overlapping.

    Args:
        agents:
        x: Position of agent that is tested
        r: Radius of agent that is tested

    Returns:
        bool:
    """
    for agent in agents:
        h, _ = distance_circles(agent['position'], agent['radius'], x, r)
        if h < 0.0:
            return True
    return False


@numba.jit([boolean(typeof(agent_type_three_circle)[:],
                    UniTuple(float64[:], 3), UniTuple(float64, 3))],
           nopython=True, nogil=True, cache=True)
def overlapping_three_circles(agents, x, r):
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
        h, _, _, _ = distance_three_circles(
            (agent['position'], agent['position_ls'], agent['position_rs']),
            (agent['r_t'], agent['r_s'], agent['r_s']),
            x, r
        )
        if h < 0:
            return True
    return False


@numba.jit([boolean(typeof(agent_type_circular)[:],
                    typeof(obstacle_type_linear)[:])],
           nopython=True, nogil=True, cache=True)
def overlapping_circle_line(agents, obstacles):
    for agent in agents:
        for obstacle in obstacles:
            h, _ = distance_circle_line(agent['position'], agent['radius'],
                                        obstacle['p0'], obstacle['p1'])
            if h < 0.0:
                return True
    return False


@numba.jit([boolean(typeof(agent_type_three_circle)[:],
                    typeof(obstacle_type_linear)[:])],
           nopython=True, nogil=True, cache=True)
def overlapping_three_circle_line(agents, obstacles):
    for agent in agents:
        for obstacle in obstacles:
            h, _, _ = distance_three_circle_line(
                (agent['position'], agent['position_ls'], agent['position_rs']),
                (agent['r_t'], agent['r_s'], agent['r_s']),
                obstacle['p0'],
                obstacle['p1']
            )
            if h < 0.0:
                return True
    return False


def call(value):
    """Iterate, call and return the value.
    
    Args:
        value:
            - Iterator: returns next(value)
            - Callable: return value()
            - otherwise returns the value itself

    Returns:

    """
    try:
        return next(value)
    except TypeError:
        pass

    try:
        return value()
    except TypeError:
        pass

    return value


def set_agent_attributes(agents, index, attributes):
    """Set attributes for agent
    
    Args:
        agents: 
        attributes:
            Dictionary values, iterators or callables. 
            - Iterator; return value of next is used
            - Callable: return value of __call__ will be used 

    """
    for attribute, value in attributes.items():
        try:
            agents[index][attribute] = call(value)
        except ValueError:
            # logger = logging.getLogger(__name__)
            # logger.warning('Agent: {} doesn\'t have attribute {}'.format(
            #     of_model(agents), attribute
            # ))
            pass


def reset_agent(index, agents):
    """Reset agent"""
    agents[index] = np.zeros(1, agents.dtype)


class Agents(object):
    """Class for initialising new agents."""
    logger = logging.getLogger(__name__)

    def __init__(self,
                 size,
                 agent_type=agent_type_circular,
                 agent_cfg=AGENT_CFG,
                 agent_cfg_spec=AGENT_CFG_SPEC):
        """Agent manager
        
        Args:
            size (int):
                Number of agents
            agent_type (str|numpy.dtype): 
                AgentModelToType
            agent_cfg:
                Agent configuration filepath
            agent_cfg_spec: 
                Agent configuration spec filepath
        """
        if isinstance(agent_type, str):
            dtype = AgentModelToType[agent_type]
        else:
            dtype = agent_type

        self.array = np.zeros(size, dtype=dtype)
        self.config = load_config(infile=agent_cfg, configspec=agent_cfg_spec)

        # Keeps track of which agents are active and which in active. Stores
        # indices of agents.
        self.active = SortedSet()
        self.inactive = SortedSet(range(size))

        # Faster check for neighbouring agents for initializing agents into
        # random positions.
        constants = self.config['constants']
        self.grid = MutableBlockList(cell_size=constants['cell_size'])

    @property
    def size(self):
        return self.array.size

    def add(self, attributes, check_overlapping=True):
        """Add new agent with given attributes

        Args:
            check_overlapping (bool): 

            attributes:
                body_type (str):
                position (numpy.ndarray): 
                    Initial position of the agent
                orientation (float): 
                    Initial orientation :math:`\varphi = [\-pi, \pi]` of the agent.
                velocity (numpy.ndarray): 
                    Initial velocity
                angular_velocity (float):
                    Angular velocity
                target_direction (numpy.ndarray): 
                    Unit vector to desired direction
                target_orientation:
                
        Raises:
            AgentStructureFull: When no more agents can be added.
            OverlappingError: When two agents overlap each other.
        """
        try:
            index = self.inactive.pop(0)
        except IndexError:
            raise AgentStructureFull

        attrs = dict(**self.config['defaults'])
        attrs.update(attributes)

        try:
            body_type = call(attrs['body_type'])
            body = self.config['body_types'][body_type]
            attrs.update(body_to_values(body))
            del attrs['body_type']
        except KeyError:
            pass

        set_agent_attributes(self.array, index, attrs)

        # Update shoulder positions for three circle agents
        if is_model(self.array, 'three_circle'):
            shoulders(self.array)

        agent = self.array[index]
        radius = agent['radius']
        position = agent['position']

        if check_overlapping:
            neighbours = self.grid.nearest(position, radius=1)
            agents = self.array[neighbours]

            if is_model(self.array, 'circular'):
                if overlapping_circles(agents, position, radius):
                    reset_agent(index, self.array)
                    self.inactive.add(index)
                    raise OverlappingError
            elif is_model(self.array, 'three_circle'):
                if overlapping_three_circles(
                        agents,
                        (agent['position'], agent['position_ls'],
                         agent['position_rs']),
                        (agent['r_t'], agent['r_s'], agent['r_s'])):
                    reset_agent(index, self.array)
                    self.inactive.add(index)
                    raise OverlappingError
            else:
                raise CrowdDynamicsException

        self.grid[position] = index
        self.active.add(index)
        return index

    @log_with(qualname=True, timed=True, ignore=('self',))
    def remove(self, index):
        """Remove agent"""
        if index in self.active:
            self.active.remove(index)
            self.inactive.add(index)
            reset_agent(index, self.array)
            return True
        else:
            return False

    @log_with(qualname=True, timed=True, ignore=('self',))
    def fill(self, amount, attributes, check_overlapping=True):
        """Fill agents

        Args:
            check_overlapping: 
            amount (int): 
            attributes (dict|Callable[dict]): 
        """
        overlaps = 0
        iteration = 0
        iterations_max = 10 * amount

        # Progress bar for displaying number of agents placed
        # FIXME: tqdm does not work with tests
        # success = tqdm(desc='Agents placed: ', total=amount)
        # overlapping = tqdm(desc='Agent overlaps: ', total=iterations_max)
        while iteration < amount and overlaps < iterations_max:
            a = attributes() if callable(attributes) else attributes
            try:
                index = self.add(a, check_overlapping=check_overlapping)
                # success.update(1)
            except OverlappingError:
                overlaps += 1
                # overlapping.update(1)
                continue
            except AgentStructureFull:
                break
            iteration += 1
        # success.close()
        # overlapping.close()

        if is_model(self.array, 'three_circle'):
            shoulders(self.array)
            front(self.array)
