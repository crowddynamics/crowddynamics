import numpy as np
import pytest

from crowddynamics.core.structures.agents import AgentManager, AgentModels, \
    reset_motion, shoulders, front, overlapping_circle_circle, \
    overlapping_three_circle, create_random_agent_attributes

SEED = np.random.randint(0, 100)
np.random.seed(SEED)


@pytest.fixture(scope='module')
def agent_circular(size=1000):
    agent = AgentManager(size, AgentModels.CIRCULAR)
    agent.fill(size, create_random_agent_attributes)
    return agent


@pytest.fixture(scope='module')
def agent_three_circle(size=1000):
    agent = AgentManager(size, AgentModels.THREE_CIRCLE)
    agent.fill(size, create_random_agent_attributes)
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
    overlapping_circle_circle(agent_circular.agents, x, r)
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
    overlapping_three_circle(agent_three_circle.agents, x, r)
    assert True
