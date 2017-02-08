from crowddynamics.multiagent.examples import Outdoor, Hallway, Rounding, \
    RoomEvacuation
from crowddynamics.multiagent.tests.test_simulation import models


def test_outdoor():
    for model in models:
        simulation = Outdoor()
        simulation.set(10, 10, 10, model, 'adult')
        simulation.update()
        simulation.update()
        assert True


def test_hallway():
    for model in models:
        simulation = Hallway()
        simulation.set(10, 10, 10, model, 'adult')
        simulation.update()
        simulation.update()
        assert True


def test_rounding():
    for model in models:
        simulation = Rounding()
        simulation.set(10, 10, 10, model, 'adult')
        simulation.update()
        simulation.update()
        assert True


def test_roomevacuation():
    for model in models:
        simulation = RoomEvacuation()
        simulation.set(10, 10, 10, model, 'adult', 'circ', 1.2, 1.5)
        simulation.update()
        simulation.update()
        assert True