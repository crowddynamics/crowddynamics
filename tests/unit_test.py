import importlib
import unittest
from multiprocessing import Queue

from typing import Iterator

from crowddynamics.functions import load_config
from crowddynamics.multiagent.simulation import MultiAgentSimulation


def import_simulations(queue: Queue=None) -> Iterator[MultiAgentSimulation]:
    """
    Yield all simulations from examples.py
    """
    configs = load_config("simulations.yaml")
    conf = configs["simulations"]
    for name in conf.keys():
        d = conf[name]
        module = importlib.import_module(d["module"])
        simulation = getattr(module, d["class"])
        process = simulation(queue, **d["kwargs"])
        yield process


class MyTestCase(unittest.TestCase):
    def test_simulations(self):
        gen = import_simulations()
        for simulation in gen:
            simulation.initial_update()
            simulation.update()
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
