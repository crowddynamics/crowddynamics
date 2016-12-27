import importlib

import pytest

from crowddynamics.functions import load_config


def simulations(queue=None):
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


@pytest.mark.skip
def test_simulation():
    for simulation in simulations():
        # simulation.configure_hdfstore()
        simulation.initial_update()
        simulation.update()
        assert True
