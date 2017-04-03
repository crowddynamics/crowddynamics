from shapely.geometry import Polygon

from crowddynamics.core.agent.agents import AgentModels
from crowddynamics.simulation.multiagent import MultiAgentSimulation


def test_field():
    height = 10
    width = 10
    size = 10
    surface = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
    body_type = 'adult'

    for model in AgentModels:
        field = MultiAgentSimulation()
        field.init_domain(None)
        field.init_agents(size, model.value)
        for i in field.add_agents(size, surface, body_type):
            assert 0 <= i < field.agent.size
            assert field.agent.active[i]
