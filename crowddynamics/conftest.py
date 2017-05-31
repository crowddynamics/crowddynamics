import random

import numpy as np
import pytest

from crowddynamics.core.vector2D import unit_vector
from crowddynamics.simulation.agents import Agents, Circular, ThreeCircle, \
    AgentGroup


def attributes():
    """Create random agent attributes"""
    orientation = np.random.uniform(-np.pi, np.pi)
    return dict(body_type=random.choice(('adult', 'male', 'female', 'child',
                                         'eldery')),
                orientation=orientation,
                velocity=np.random.uniform(0.0, 1.0, 2),
                angular_velocity=np.random.uniform(-1.0, 1.0),
                target_direction=unit_vector(orientation),
                target_orientation=orientation)


@pytest.fixture(scope='function')
def agents_circular(size=10):
    agents = Agents(agent_type=Circular)
    group = AgentGroup(agent_type=Circular, size=size, attributes=attributes)
    agents.add_non_overlapping_group(
        group, position_gen=lambda: np.random.uniform(-10.0, 10.0, 2))
    return agents


@pytest.fixture(scope='function')
def agents_three_circle(size=10):
    agents = Agents(agent_type=ThreeCircle)
    group = AgentGroup(agent_type=ThreeCircle, size=size, attributes=attributes)
    agents.add_non_overlapping_group(
        group, position_gen=lambda: np.random.uniform(-10.0, 10.0, 2))
    return agents
