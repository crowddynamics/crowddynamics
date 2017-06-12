import random
from collections import Collection, Generator, Callable

import numpy as np
import pytest

from crowddynamics.core.vector2D import unit_vector
from crowddynamics.simulation.agents import (
    Circular, ThreeCircle, AgentGroup, Agents,
    AgentType, overlapping_circles,
    overlapping_three_circles)

SIZE = 10
XMIN = -10
XMAX = 10
SEED = np.random.randint(0, 100)
np.random.seed(SEED)


def random_attributes():
    """Create random agent attributes"""
    body_types = ('adult', 'male', 'female', 'child', 'eldery')
    orientation = np.random.uniform(-np.pi, np.pi)
    return {
        'body_type': random.choice(body_types),
        'position': np.random.uniform(XMIN, XMAX, 2),
        'orientation': orientation,
        'velocity': np.random.uniform(0.0, 1.0, 2),
        'angular_velocity': np.random.uniform(-1.0, 1.0),
        'target_direction': unit_vector(orientation),
        'target_orientation': orientation
    }


@pytest.mark.parametrize('agent_type, attributes', [
    (Circular, random_attributes),
    (ThreeCircle, random_attributes)
])
def test_agent_types(agent_type, attributes):
    agent = agent_type(**attributes())
    assert True


@pytest.mark.parametrize('agent_type, attributes', [
    (Circular, random_attributes),
    (ThreeCircle, random_attributes)
])
def test_agent_group(agent_type, attributes):
    group = AgentGroup(size=SIZE, agent_type=agent_type, attributes=attributes)

    assert isinstance(group.size, int)
    assert issubclass(group.agent_type, AgentType)
    assert isinstance(group.attributes, (Collection, Generator, Callable))

    assert isinstance(group.members, list)
    assert len(group.members) == SIZE


@pytest.mark.parametrize('agent_type, attributes', [
    (Circular, random_attributes),
    (ThreeCircle, random_attributes)
])
def test_agents(agent_type, attributes):
    agents = Agents(agent_type=agent_type)
    group = AgentGroup(size=SIZE, agent_type=agent_type, attributes=attributes)
    agents.add_non_overlapping_group(
        group=group,
        position_gen=lambda: np.random.uniform(XMIN, XMAX, 2))
    assert True


def test_overlapping_circular(agents_circular):
    x = np.random.uniform(-1.0, 1.0, 2)
    r = np.random.uniform(0.0, 1.0)
    overlapping_circles(agents_circular.array, x, r)
    assert True


def test_overlapping_three_circle(agents_three_circle):
    x = (
        np.random.uniform(-1.0, 1.0, 2),
        np.random.uniform(-1.0, 1.0, 2),
        np.random.uniform(-1.0, 1.0, 2)
    )
    r = (
        np.random.uniform(0.0, 1.0),
        np.random.uniform(0.0, 1.0),
        np.random.uniform(0.0, 1.0)
    )
    overlapping_three_circles(agents_three_circle.array, x, r)
    assert True
