import numpy as np
import pytest

from crowddynamics.core.interactions import agent_agent_block_list
from crowddynamics.core.vector2D import unit_vector
from crowddynamics.simulation.agents import Agents, Circular, ThreeCircle, \
    AgentGroup


def attributes():
    orientation = np.random.uniform(-np.pi, np.pi)
    return dict(body_type='adult',
                orientation=orientation,
                velocity=np.random.uniform(0.0, 1.3, 2),
                angular_velocity=np.random.uniform(-1.0, 1.0),
                target_direction=unit_vector(orientation),
                target_orientation=orientation)


@pytest.mark.parametrize('size', (200, 500, 1000))
@pytest.mark.parametrize('agent_type', (Circular, ThreeCircle))
def test_agent_agent_block_list(benchmark, size, agent_type, algorithm):
    # Grow the area with size. Keeps agent density constant.
    area_size = np.sqrt(2 * size)
    agents = Agents(agent_type=agent_type)
    group = AgentGroup(
        agent_type=agent_type,
        size=size,
        attributes=attributes)
    agents.add_non_overlapping_group(
        group, position_gen=lambda: np.random.uniform(-area_size, area_size, 2))
    benchmark(agent_agent_block_list, agents.array)
    assert True
