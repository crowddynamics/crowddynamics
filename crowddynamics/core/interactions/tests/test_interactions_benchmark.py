import pytest
import numpy as np

from crowddynamics.core.interactions.interactions import agent_agent_block_list
from crowddynamics.core.interactions.interactions_mt import \
    agent_agent_block_list_multithreaded
from crowddynamics.core.structures.agents import agent_type_circular, \
    agent_type_three_circle
from crowddynamics.simulation.multiagent import Agents
from crowddynamics.core.vector2D import unit_vector


@pytest.mark.parametrize('size', (200, 500, 1000))
@pytest.mark.parametrize('agent_type', (agent_type_circular,
                                        agent_type_three_circle))
@pytest.mark.parametrize('algorithm', (agent_agent_block_list,
                                       agent_agent_block_list_multithreaded))
def test_agent_agent_block_list(benchmark, size, agent_type, algorithm):
    # Grow the area with size. Keeps agent density constant.
    area_size = np.sqrt(2 * size)
    agents = Agents(size, agent_type)
    agents.add_group(size, {
        'body_type': 'adult',
        'position': lambda: np.random.uniform(-area_size, area_size, 2),
        'orientation': lambda: np.random.uniform(-np.pi, np.pi),
        'velocity': lambda: np.random.uniform(0.0, 1.3, 2),
        'angular_velocity': lambda: np.random.uniform(-1.0, 1.0),
        'target_direction': lambda: unit_vector(np.random.uniform(-np.pi, np.pi)),
        'target_orientation': lambda: np.random.uniform(-np.pi, np.pi)
    })
    benchmark(algorithm, agents.array)
    assert True
