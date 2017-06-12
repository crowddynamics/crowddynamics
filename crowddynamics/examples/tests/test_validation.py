import pytest
import numpy as np

from crowddynamics.core.vector2D import length
from crowddynamics.examples.validation import TestMovement, TestAgentInteraction
from crowddynamics.simulation.agents import AgentTypes

# TODO: set dt
# dt_min = 0.01
# dt_max = 0.01


@pytest.mark.parametrize('agent_type', AgentTypes)
def test_movement(agent_type):
    simulation = TestMovement(agent_type=agent_type)
    agent_start = np.copy(simulation.agents.array)

    simulation.exit_condition = lambda s: s.data['iterations'] == 1000
    simulation.run()

    agent_end = np.copy(simulation.agents.array)
    dist = length(agent_end[0]['position'] - agent_start[0]['position'])
    expected_dist = 10.0
    assert dist >= expected_dist or np.isclose(dist, expected_dist)


@pytest.mark.parametrize('agent_type', AgentTypes)
def test_agent_interaction(agent_type):
    simulation = TestAgentInteraction(agent_type=agent_type)
    agent_start = np.copy(simulation.agents.array)

    simulation.exit_condition = lambda s: s.data['iterations'] == 1000
    simulation.run()

    agent_end = np.copy(simulation.agents.array)
    dist = length(agent_end[0]['position'] - agent_start[0]['position'])
    expected_dist = 8.0
    assert dist >= expected_dist or np.isclose(dist, expected_dist)
