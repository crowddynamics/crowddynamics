import numpy as np
import pytest

from crowddynamics.core.structures.agents import AgentManager, \
    reset_motion, shoulders, front, overlapping_circles, \
    agent_type_circular, agent_type_three_circle, \
    overlapping_three_circles
from crowddynamics.core.vector import unit_vector

SEED = np.random.randint(0, 100)
np.random.seed(SEED)


def create_random_agent_attributes():
    return {
        'body_type': 'adult',
        'position': np.random.uniform(-1.0, 1.0, 2),
        'orientation': np.random.uniform(-np.pi, np.pi),
        'velocity': np.random.uniform(0.0, 1.0, 2),
        'angular_velocity': np.random.uniform(-1.0, 1.0),
        'target_direction': unit_vector(np.random.uniform(-np.pi, np.pi)),
        'target_orientation': np.random.uniform(-np.pi, np.pi)
    }


@pytest.fixture(scope='module')
def agent_circular(size=100):
    agent = AgentManager(size, agent_type_circular)
    agent.fill(size, create_random_agent_attributes())
    return agent


@pytest.fixture(scope='module')
def agent_three_circle(size=100):
    agent = AgentManager(size, agent_type_three_circle)
    agent.fill(size, create_random_agent_attributes())
    return agent


def test_circular(agent_circular):
    reset_motion(agent_circular.agents)
    assert True


def test_three_circle(agent_three_circle):
    reset_motion(agent_three_circle.agents)
    shoulders(agent_three_circle.agents)
    front(agent_three_circle.agents)
    assert True


def test_overlapping_circular(agent_circular):
    x = np.random.uniform(-1.0, 1.0, 2)
    r = np.random.uniform(0.0, 1.0)
    overlapping_circles(agent_circular.agents, x, r)
    assert True


def test_overlapping_three_circle(agent_three_circle):
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
    overlapping_three_circles(agent_three_circle.agents, x, r)
    assert True
