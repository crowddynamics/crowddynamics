r"""Agent structure"""
import numba
import numpy as np
from numba import typeof, void, boolean, float64
from numba.types import UniTuple

from crowddynamics.core.interactions.distance import distance_circles, \
    distance_circle_line, distance_three_circle_line
from crowddynamics.core.interactions.distance import distance_three_circles
from crowddynamics.core.structures.obstacles import obstacle_type_linear
from crowddynamics.core.vector2D import unit_vector, rotate270

states = [
    ('active', np.bool_),
]
navigation = [
    ('target', np.uint8),
    ('target_reached', np.bool_),
]
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
]

agent_type_circular = np.dtype(
    states +
    navigation +
    translational
)
agent_type_three_circle = np.dtype(
    states +
    navigation +
    translational +
    rotational +
    three_circle
)


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
