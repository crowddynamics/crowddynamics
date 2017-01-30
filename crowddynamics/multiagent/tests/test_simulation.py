from crowddynamics.multiagent.examples import Outdoor, Hallway, Rounding, \
    RoomEvacuation
from crowddynamics.multiagent.parameters import Parameters

parameters = Parameters()
models = parameters.model.values
body_types = parameters.body_types


def test_outdoor():
    for model in models:
        simulation = Outdoor(None, 10, 10, 10, model, 'adult')
        simulation.initial_update()
        simulation.update()
        assert True


def test_hallway():
    for model in models:
        simulation = Hallway(None, 10, 10, 10, model, 'adult')
        simulation.initial_update()
        simulation.update()
        assert True


def test_rounding():
    for model in models:
        simulation = Rounding(None, 10, 10, 10, model, 'adult')
        simulation.initial_update()
        simulation.update()
        assert True


def test_roomevacuation():
    for model in models:
        simulation = RoomEvacuation(None, 10, 10, 10, model, 'adult',
                                    'circ', 1.2, 1.5)
        simulation.initial_update()
        simulation.update()
        assert True
