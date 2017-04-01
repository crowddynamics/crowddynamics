import argparse
import logging
import logging.config
import os
import platform
import sys

from ruamel import yaml

from crowddynamics.functions import numpy_format, \
    pandas_format

LOG_CFG = 'configs/logging.yaml'


def setup_logging(default_path=LOG_CFG,
                  default_level=logging.INFO,
                  env_key='LOG_CFG'):
    """
    Setup logging configuration
    """

    # TODO: Change root logger -> custom logger
    # Path to logging yaml configuration file.
    filepath = os.path.abspath(__file__)
    folderpath = os.path.split(filepath)[0]
    path = os.path.join(folderpath, default_path)

    # Set-up logging
    logger = logging.getLogger(__name__)
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

    # Nicer printing for numpy array and pandas tables
    numpy_format()
    pandas_format()


def gui():
    """
    Parses command line arguments, setups logging functionality and launches
    graphical user interface for visualizing simulation.
    """
    logger = logging.getLogger("crowddynamics.gui.mainwindow")

    # Qt - Graphical User Interface
    logger.info("Starting GUI")
    from PyQt4 import QtGui, QtCore
    from crowddynamics.gui.main import MainWindow

    app = QtGui.QApplication(sys.argv)
    win = MainWindow()
    win.show()

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app.exec()
    else:
        logger.warning("Interactive mode and pyside are not supported.")

    logger.info("Finishing GUI\n")
    logging.shutdown()

    win.close()
    app.exit()
    sys.exit()


def main(run_gui=True):
    """
    Parse command line arguments
    """
    # Logging arguments
    parser = argparse.ArgumentParser(description="")
    loglevels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    parser.add_argument("-l", "--log",
                        dest="logLevel",
                        choices=loglevels,
                        help="Set the logging level")

    # Parse arguments
    args = parser.parse_args()
    if args.logLevel:
        log_level = args.logLevel
    else:
        log_level = logging.INFO

    setup_logging(default_level=log_level)

    # User info
    logger = logging.getLogger("crowddynamics")
    logger.info("Platform: %s", platform.platform())
    logger.info("Path: %s", sys.path[0])
    logger.info("Python: %s", sys.version[0:5])

    if run_gui:
        gui()
    else:
        # TODO: Run only simulation
        raise NotImplementedError
