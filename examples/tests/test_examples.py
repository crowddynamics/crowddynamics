import pytest
from crowddynamics.core.structures.agents import AgentModelToType

from examples.examples import outdoor, hallway, rounding, room_evacuation, \
    uturn


@pytest.mark.parametrize('simulation',
                         (outdoor, hallway, rounding, room_evacuation, uturn))
@pytest.mark.parametrize('agent_type', tuple(AgentModelToType.values()))
def test_simulations(simulation, agent_type):
    simu = simulation(agent_type=agent_type)
    simu.update()
