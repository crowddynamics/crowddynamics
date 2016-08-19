import argparse
import logging
import logging.config
import os
import platform
import sys

import yaml


def man():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-l", "--log",
                        dest="logLevel",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR',
                                 'CRITICAL'],
                        help="Set the logging level")
    return parser


def user_info():
    logging.info("Platform: %s", platform.platform())
    logging.info("Path: %s", sys.path[0])
    logging.info("Python: %s", sys.version[0:5])


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


def run_gui():
    """Launches Qt application for visualizing simulation.
    :param simulation:
    """
    from PyQt4 import QtGui, QtCore
    sys.path.insert(0, os.path.abspath(".."))
    from src.Qt.main import MainWindow

    args = man().parse_args()
    if args.logLevel:
        log_level = args.logLevel
    else:
        log_level = logging.INFO

    setup_logging(default_level=log_level)
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

    logging.info("Finishing")
    logging.shutdown()

    win.close()
    app.exit()
    sys.exit()
