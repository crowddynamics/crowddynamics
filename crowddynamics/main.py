import argparse
import importlib
import logging
import logging.config
import os
import platform
import sys

from ruamel import yaml


from crowddynamics.config import Load


def man():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-l", "--log",
                        dest="logLevel",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR',
                                 'CRITICAL'],
                        help="Set the logging level")
    return parser


def setup_logging(default_path='configs/logging.yaml',
                  default_level=logging.INFO,
                  env_key='LOG_CFG'):
    """Setup logging configuration

    Logging levels
    --------------
    CRITICAL 50
    ERROR    40
    WARNING  30
    INFO     20
    DEBUG    10
    NOTSET    0
    """
    try:
        import numpy as np
        np.set_printoptions(precision=5, threshold=6, edgeitems=3, linewidth=None,
                            suppress=False, nanstr=None, infstr=None,
                            formatter=None)
    except ImportError():
        pass

    try:
        import pandas as pd
        pandas_options = {'display.chop_threshold': None,
                          'display.precision': 4,
                          'display.max_columns': 8,
                          'display.max_rows': 8,
                          'display.max_info_columns': 8,
                          'display.max_info_rows': 8}
        for key, val in pandas_options.items():
            pd.set_option(key, val)
    except ImportError():
        pass

    filepath = os.path.abspath(__file__)
    folderpath = os.path.split(filepath)[0]
    path = os.path.join(folderpath, default_path)
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def user_info():
    logging.info("Platform: %s", platform.platform())
    logging.info("Path: %s", sys.path[0])
    logging.info("Python: %s", sys.version[0:5])


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

    process.initial_update()
    for _ in range(iterations):
        process.update()


def run_gui():
    """
    Parses command line arguments, setups logging functionality and launches
    graphical user interface for visualizing simulation.
    """
    from PyQt4 import QtGui, QtCore
    sys.path.insert(0, os.path.abspath(".."))
    from crowddynamics.gui.main import MainWindow

    args = man().parse_args()
    if args.logLevel:
        log_level = args.logLevel
    else:
        log_level = logging.INFO

    setup_logging()
    user_info()
    logging.info("Starting")

    app = QtGui.QApplication(sys.argv)
    win = MainWindow()
    win.show()

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app.exec()
    else:
        logging.warning("Interactive mode or pyside are not supported.")

    logging.info("Finishing\n")
    logging.shutdown()

    win.close()
    app.exit()
    sys.exit()
