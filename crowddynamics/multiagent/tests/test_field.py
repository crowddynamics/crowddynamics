from shapely.geometry import Polygon

from crowddynamics.multiagent import Field
from crowddynamics.multiagent.parameters import Parameters

models = ('circular', 'three_circle')
body_types = Parameters().body_types


def test_field():
    height = 10
    width = 10
    size = 10
    surface = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
    body_type = 'adult'

    for model in models:
        field = Field()
        field.init_domain(None)
        field.init_agents(size, model)
        for i in field.add_agents(size, surface, body_type):
            assert 0 <= i < field.agent.size
            assert field.agent.active[i]
