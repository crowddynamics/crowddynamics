import logging
import sys

from PyQt4 import QtGui, QtCore

from crowddynamics.logging import log_with
from crowddynamics.plugins.gui.main import MainWindow

logger = logging.getLogger(__name__)


@log_with(logger, entry_msg="Starting GUI", exit_msg='Finishing GUI\n')
def run_gui():
    r"""Launches the graphical user interface for visualizing simulation."""
    app = QtGui.QApplication(sys.argv)
    win = MainWindow()
    win.show()

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app.exec()
    else:
        logger.warning("Interactive mode and pyside are not supported.")

    # logging.shutdown()
    win.close()
    app.exit()
    sys.exit()
