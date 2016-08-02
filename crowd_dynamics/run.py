import sys

from PyQt4 import QtGui, QtCore

from crowd_dynamics.Qt.qui import Gui
from crowd_dynamics.simulation import Simulation

sys.path.append("/home/jaan/Dropbox/Projects/Crowd-Dynamics")

path = "/home/jaan/Dropbox/Projects/Crowd-Dynamics-Simulations/"
path2 = "/media/storage3/Crowd-Dynamics-Simulations/"


def outdoor(density):
    from crowd_dynamics.examples.outdoor import initialize
    kwargs = {}
    if density == "low":
        kwargs.update(size=30, width=10, height=10, path=path)
    elif density == "medium":
        kwargs.update(size=100, width=10, height=10, path=path)
    elif density == "high":
        kwargs.update(size=200, width=10, height=10, path=path)
    return initialize(**kwargs)


def hallway(density):
    from crowd_dynamics.examples.hallway import initialize
    kwargs = {}
    if density == "low":
        kwargs.update(size=30, width=30, height=5, path=path)
    elif density == "medium":
        kwargs.update(size=100, width=30, height=5, path=path)
    elif density == "high":
        kwargs.update(size=200, width=30, height=5, path=path)
    return initialize(**kwargs)


def evacuation(density):
    from crowd_dynamics.examples.evacuation import initialize
    kwargs = {}
    if density == "low":
        kwargs.update(size=30, width=5, height=10, path=path)
    elif density == "medium":
        kwargs.update(size=200, width=10, height=20, egress_model=True,
                      t_aset=45, path=path)
    elif density == "high":
        kwargs.update(size=700, width=15, height=30, egress_model=True,
                      t_aset=250, path=path2, name="evacuation2_700")
    return initialize(**kwargs)


def main(simulation: Simulation=None):
    """Launches Qt application for visualizing simulation.
    :param simulation:
    """
    # TODO: Read simulation data from hdf5 file
    # TODO: MoviePy
    import sys

    app = QtGui.QApplication(sys.argv)
    qui = Gui(simulation)

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(app.exec_())
    else:
        raise Warning("Interactive mode or Pyside are not supported.")


if __name__ == '__main__':
    # TODO: Better agent initialization.
    # TODO: Circular agent selection
    # TODO: Polygonal chain, non overlapping, radius/thickness
    # TODO: Agent maximum velocity for optimizations
    # simulation = outdoor("medium")
    # simulation = hallway("medium")
    # simulation = evacuation("medium")
    # main(simulation)
    main()
