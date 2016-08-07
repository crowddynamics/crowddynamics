import sys

from PyQt4 import QtGui, QtCore

sys.path.append("/home/jaan/Dropbox/Projects/Crowd-Dynamics")


def run_gui():
    """Launches Qt application for visualizing simulation.
    :param simulation:
    """
    from src.Qt.main import MainWindow

    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(app.exec_())
    else:
        raise Warning("Interactive mode or pyside are not supported.")


if __name__ == '__main__':
    # TODO: Better agent initialization.
    # TODO: Polygonal chain, non overlapping, radius/thickness
    # TODO: Agent maximum velocity for optimizations
    # TODO: Read simulation data from hdf5 file
    # TODO: MoviePy
    run_gui()
