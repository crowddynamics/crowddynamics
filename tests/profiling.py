import importlib
import logging
import logging.config
import sys

sys.path.insert(0, "..")

from crowddynamics.main import setup_logging
from crowddynamics.functions import load_config


def run_simulation(name, iterations=100):
    setup_logging()
    logging.info("Starting")

    configs = load_config("simulations.yaml")
    simu_dict = configs["simulations"][name]
    module_name = simu_dict["module"]
    class_name = simu_dict["class"]
    kwargs = simu_dict["kwargs"]

    module = importlib.import_module(module_name)
    simulation = getattr(module, class_name)
    process = simulation(None, **kwargs)

    # process.configure_hdfstore()

    process.initial_update()
    for _ in range(iterations):
        process.update()
