import sys; sys.path.insert(0, "..")
import importlib
import unittest

from multiprocessing import Queue

from typing import Iterator

from crowddynamics.multiagent.simulation import MultiAgentSimulation
from crowddynamics.functions import load_config


def import_simulations(queue: Queue=None) -> Iterator[MultiAgentSimulation]:
    """
    Yield all simulations from examples.py
    """
    configs = load_config("simulations.yaml")
    d = configs["simulations"]
    for name in d.keys():
        d = d[name]
        module = importlib.import_module(d["module"])
        simulation = getattr(module, d["class"])
        process = simulation(queue, **d["kwargs"])
        yield process


class MyTestCase(unittest.TestCase):
    def test_attributes(self):
        self.assertTrue(True)

    def test_simulations(self):
        gen = import_simulations()
        for simulation in gen:
            pass


if __name__ == '__main__':
    unittest.main()
