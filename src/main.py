import argparse
import logging as log
import os
import sys

from src.log import start_logging, user_info


def man():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-l", "--log",
                        dest="logLevel",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level")
    return parser


def run_gui():
    """Launches Qt application for visualizing simulation.
    :param simulation:
    """
    from PyQt4 import QtGui, QtCore
    sys.path.insert(0, os.path.abspath(".."))
    from src.Qt.main import MainWindow

    filename = "run"
    args = man().parse_args()
    if args.logLevel:
        start_logging(args.logLevel, filename)
    else:
        start_logging(log.INFO, filename)
    user_info()

    name = "CrowdDynamics"
    log.info("Starting {name}".format(name=name))

    app = QtGui.QApplication(sys.argv)
    win = MainWindow()
    win.show()

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app.exec()
    else:
        log.warning("Interactive mode or pyside are not supported.")

    log.info("Finishing {name}\n".format(name=name))
    log.shutdown()

    win.close()
    app.exit()
    sys.exit()
