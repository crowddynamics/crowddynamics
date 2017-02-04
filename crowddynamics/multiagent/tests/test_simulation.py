from crowddynamics.multiagent.examples import Outdoor, Hallway, Rounding, \
    RoomEvacuation
from crowddynamics.multiagent.parameters import Parameters

from shapely.geometry import Polygon

from crowddynamics.multiagent.simulation import MultiAgentSimulation

parameters = Parameters()
models = parameters.model.values
body_types = parameters.body_types


def test_field():
    height = 10
    width = 10
    size = 10
    surface = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
    body_type = 'adult'

    for model in models:
        field = MultiAgentSimulation()
        field.init_domain(None)
        field.init_agents(size, model)
        for i in field.add_agents(size, surface, body_type):
            assert 0 <= i < field.agent.size
            assert field.agent.active[i]


def test_outdoor():
    for model in models:
        simulation = Outdoor(10, 10, 10, model, 'adult')
        simulation.update()
        simulation.update()
        assert True


def test_hallway():
    for model in models:
        simulation = Hallway(10, 10, 10, model, 'adult')
        simulation.update()
        simulation.update()
        assert True


def test_rounding():
    for model in models:
        simulation = Rounding(10, 10, 10, model, 'adult')
        simulation.update()
        simulation.update()
        assert True


def test_roomevacuation():
    for model in models:
        simulation = RoomEvacuation(10, 10, 10, model, 'adult', 'circ', 1.2,
                                    1.5)
        simulation.update()
        simulation.update()
        assert True
