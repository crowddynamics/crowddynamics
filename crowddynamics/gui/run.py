import logging
import sys

from PyQt4 import QtGui, QtCore

from crowddynamics.gui.main import MainWindow


def run_gui():
    """Launches the graphical user interface for visualizing simulation."""
    logger = logging.getLogger(__name__)

    # Qt - Graphical User Interface
    logger.info("Starting GUI")

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
