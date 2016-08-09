import argparse
import logging as log
import logging.handlers
import os
import platform
import sys

from PyQt4 import QtGui, QtCore


def start_logging(level):
    log_format = log.Formatter('%(asctime)s, %(levelname)s, %(message)s')
    logger = log.getLogger()
    logger.setLevel(level)

    file_handler = logging.handlers.RotatingFileHandler("run.log",
                                                        maxBytes=(10240 * 5),
                                                        backupCount=2)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    console_handler = log.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)


def user_info():
    log.info("Platform: %s", platform.platform())
    log.info("Path: %s", sys.path[0])
    log.info("Python: %s", sys.version[0:5])


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
    sys.path.insert(0, os.path.abspath(".."))
    from src.Qt.main import MainWindow

    args = man().parse_args()
    if args.logLevel:
        start_logging(args.logLevel)
    else:
        start_logging(log.INFO)
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


# TODO: Better agent initialization.
# TODO: Polygonal chain, non overlapping, radius/thickness
# TODO: Agent maximum velocity for optimizations
# TODO: Read simulation data from hdf5 file
# TODO: MoviePy
# TODO: Logger
# TODO: Multiprocessor
# TODO: density, navigation visualization
# TODO: Frames per second
# TODO: Saving and loading HDF5.
# TODO: start/end time, simulation length
# TODO: Bytes saved, memory consumption


if __name__ == '__main__':
    run_gui()
