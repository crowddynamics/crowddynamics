import pytest

from crowddynamics.core.agent.agents import AgentManager, AgentModels, \
    AgentBodyTypes


@pytest.fixture()
def agent_circular():
    return AgentManager(10, AgentModels.CIRCULAR)


@pytest.fixture()
def agent_three_circle():
    return AgentManager(10, AgentModels.THREE_CIRCLE)


def test_circular(agent_circular):
    assert True


def test_three_circle():
    assert True


def test_linear_obstacle():
    assert True


def test_neighborhood():
    assert True
