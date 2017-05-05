import random

import numpy as np
import pytest

from crowddynamics.core.structures.agents import agent_type_circular, \
    agent_type_three_circle
from crowddynamics.simulation.multiagent import Agents
from crowddynamics.core.vector2D import unit_vector


def _random_attributes():
    """Create random agent attributes"""
    return {
        'body_type': lambda: random.choice(
            ('adult', 'male', 'female', 'child', 'eldery')),
        'position': lambda: np.random.uniform(-1.0, 1.0, 2),
        'orientation': lambda: np.random.uniform(-np.pi, np.pi),
        'velocity': lambda: np.random.uniform(0.0, 1.0, 2),
        'angular_velocity': lambda: np.random.uniform(-1.0, 1.0),
        'target_direction': lambda: unit_vector(np.random.uniform(-np.pi, np.pi)),
        'target_orientation': lambda: np.random.uniform(-np.pi, np.pi)
    }


@pytest.fixture(scope='function')
def agents_circular(size=10):
    agents = Agents(size, agent_type_circular)
    agents.add_group(size, _random_attributes())
    return agents


@pytest.fixture(scope='function')
def agents_three_circle(size=10):
    agents = Agents(size, agent_type_three_circle)
    agents.add_group(size, _random_attributes())
    return agents


@pytest.fixture(scope='function')
def linear_walls():
    pass
