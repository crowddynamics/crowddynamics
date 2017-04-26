import pytest
import numpy as np

from crowddynamics.core.interactions.interactions import agent_agent_block_list
from crowddynamics.core.structures.agents import Agents, agent_type_circular, \
    agent_type_three_circle
from crowddynamics.core.vector import unit_vector


@pytest.mark.parametrize('size', (200, 500, 1000))
@pytest.mark.parametrize('agent_type', (agent_type_circular,
                                        agent_type_three_circle))
def test_agent_agent_block_list(benchmark, size, agent_type):
    # Grow the area with size. Keeps agent density constant.
    area_size = np.sqrt(2 * size)
    agents = Agents(size, agent_type)
    agents.fill(size, {
        'body_type': 'adult',
        'position': lambda: np.random.uniform(-area_size, area_size, 2),
        'orientation': lambda: np.random.uniform(-np.pi, np.pi),
        'velocity': lambda: np.random.uniform(0.0, 1.3, 2),
        'angular_velocity': lambda: np.random.uniform(-1.0, 1.0),
        'target_direction': lambda: unit_vector(np.random.uniform(-np.pi, np.pi)),
        'target_orientation': lambda: np.random.uniform(-np.pi, np.pi)
    })
    benchmark(agent_agent_block_list, agents.array)
    assert True
