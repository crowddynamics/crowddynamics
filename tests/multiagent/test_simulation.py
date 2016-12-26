import importlib
import unittest

from crowddynamics.functions import load_config


def import_simulations(queue=None):
    """
    Yield all simulations from examples.py

    Args:
        queue (multiprocessing.Queue, optional):

    Yields:
        MultiAgentSimulation:
    """
    configs = load_config("simulations.yaml")
    conf = configs["simulations"]
    for name in conf.keys():
        d = conf[name]
        module = importlib.import_module(d["module"])
        simulation = getattr(module, d["class"])
        process = simulation(queue, **d["kwargs"])
        yield process


class MultiAgentSimulationTest(unittest.TestCase):
    def test_simulations(self):
        gen = import_simulations()
        for simulation in gen:
            # simulation.configure_hdfstore()
            simulation.initial_update()
            simulation.update()
            self.assertTrue(True)