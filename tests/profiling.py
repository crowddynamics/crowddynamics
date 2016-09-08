"""
Profiling and Benchmarks
========================

kernprof -l profiling.py

python -m line_profiler profiling.py.lprof


Jit compiling speed


.. [#] http://www.scipy-lectures.org/advanced/optimizing/
.. [#] http://www.blog.pythonlibrary.org/2016/05/24/python-101-an-intro-to-benchmarking-your-code/

"""
import sys
import importlib
import logging
import logging.config

sys.path.insert(0, "/home/jaan/Dropbox/Projects/CrowdDynamics")

from src.main import setup_logging, user_info
from src.config import Load


def run_simulation(name, iterations=100):
    setup_logging()
    user_info()
    logging.info("Starting")

    load = Load()
    configs = load.yaml("simulations")
    simu_dict = configs["simulations"][name]
    module_name = simu_dict["module"]
    class_name = simu_dict["class"]
    kwargs = simu_dict["kwargs"]

    module = importlib.import_module(module_name)
    simulation = getattr(module, class_name)
    process = simulation(None, **kwargs)

    process.configure_hdfstore()

    process.initial_update()
    for _ in range(iterations):
        process.update()

run_simulation("room_evacuation", 500)
