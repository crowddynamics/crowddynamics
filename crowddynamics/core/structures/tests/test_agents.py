import numpy as np
import pytest

from crowddynamics.core.structures.agents import AgentManager, \
    reset_motion, shoulders, front, overlapping_circles, \
    agent_type_circular, agent_type_three_circle, \
    overlapping_three_circles

SEED = np.random.randint(0, 100)
np.random.seed(SEED)


@pytest.fixture(scope='module')
def agent_circular(size=1000):
    agent = AgentManager(size, agent_type_circular)
    agent.fill(size, {
        'body_type': 'adult',
        'position': np.random.uniform(size=2),
        'velocity': np.random.uniform(size=2),
        'target_direction': np.random.uniform(size=2),
    })
    return agent


@pytest.fixture(scope='module')
def agent_three_circle(size=1000):
    agent = AgentManager(size, agent_type_three_circle)
    agent.fill(size, {
        'body_type': 'adult',
        'position': np.random.uniform(size=2),
        'velocity': np.random.uniform(size=2),
        'target_direction': np.random.uniform(size=2),
        'orientation': np.random.uniform(-np.pi, np.pi),
    })
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
