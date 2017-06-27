import os

import pytest

from crowddynamics.simulation.agents import AgentTypes
from crowddynamics.simulation.multiagent import MultiAgentSimulation
from crowddynamics.utils import import_subclasses

root = os.path.dirname(os.path.dirname(__file__))
simulations = import_subclasses(os.path.join(root, 'simulations.py'),
                                MultiAgentSimulation)


@pytest.mark.parametrize('simulation', simulations.values())
@pytest.mark.parametrize('agent_type', AgentTypes)
def test_simulations(simulation, agent_type):
    simu = simulation(agent_type=agent_type)
    simu.exit_condition = lambda s: s.data['iterations'] == 100
    simu.run()
