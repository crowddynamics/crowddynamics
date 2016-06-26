import numpy as np
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore

from .area import GoalRectangle, Bounds
from .simulation import Simulation
from .structure.wall import LinearWall
from .structure.wall import RoundWall


class CentralItem(pg.PlotItem):
    name = "central_item"
    title = "Simulation"

    def __init__(self, simulation: Simulation):
        """
        Central plot.
        """
        # TODO: Remote processing
        # TODO: Legend
        # TODO: Coloring of agents (Forces, etc)

        super(CentralItem, self).__init__(name=self.name, title=self.title)

        # One to one scale for x and y coordinates
        self.setAspectLocked(lock=True, ratio=1)

        # Data
        self.simulation = simulation
        self.agent = simulation.agent

        # Areas. TODO: Grid, XRange, YRange
        self.bounds = self.plot()
        self.goals = self.plot()
        self.addAreas()

        # Agent. TODO: Orientable vs circular
        connect = np.ones(3 * self.agent.size, dtype=np.int32)
        connect[2::3] = np.zeros(self.agent.size, dtype=np.int32)
        self.left_shoulder = self.addCircle(self.agent.r_s)
        self.right_shoulder = self.addCircle(self.agent.r_s)
        self.center = self.addCircle(self.agent.r_t)
        self.direction = self.plot(connect=connect)
        self.updateData()

        # Walls
        self.walls = self.plot()
        self.addWalls()

    def addCircle(self, sizes):
        return self.plot(symbol='o', symbolSize=2 * sizes, pen=None, pxMode=False)

    def addWalls(self):
        for wall in self.simulation.wall:
            if isinstance(wall, LinearWall):
                connect = np.zeros(2 * wall.size, dtype=np.int32)
                connect[::2] = np.ones(wall.size, dtype=np.int32)
                self.walls.setData(wall.params[:, :, 0].flatten(),
                                   wall.params[:, :, 1].flatten(),
                                   connect=connect)
            elif isinstance(wall, RoundWall):
                self.walls.setData(wall.params[:, :2],
                                   symbolSize=wall.params[:, 2],
                                   symbol='o', pen=None, pxMode=False)

    def addAreas(self):
        # TODO: simulation.goals -> simulation.areas
        for area in self.simulation.goals:
            if isinstance(area, GoalRectangle):
                pass
            elif isinstance(area, Bounds):
                pass

    def updateData(self):
        """
        Updates data in the plot.
        """
        self.center.setData(self.agent.position)
        self.left_shoulder.setData(self.agent.position_ls)
        self.right_shoulder.setData(self.agent.position_rs)

        array = np.concatenate((self.agent.position_ls,
                                self.agent.front,
                                self.agent.position_rs), axis=1)
        array = array.reshape(3 * self.agent.shape[0], self.agent.shape[1])
        self.direction.setData(array)


class Controls(QtGui.QWidget):
    def __init__(self):
        super().__init__()


class Graphics(pg.GraphicsLayoutWidget):
    def __init__(self, simulation: Simulation, parent=None, **kargs):
        """
        Contains all the plots. Updates interactive plots.
        """
        super().__init__(parent, **kargs)

        self.simulation = simulation

        pg.setConfigOptions(antialias=True)
        self.setWindowTitle("Crowd Dynamics")
        self.resize(*(1200, 800))

        self.central = CentralItem(self.simulation)
        self.addItem(self.central, 0, 0, 1, 1)  # row, col, rowspan, colspan

        self.timer = QtCore.QTimer()
        # noinspection PyUnresolvedReferences
        self.timer.timeout.connect(self.updatePlots)
        self.timer.start(0)

    def updatePlots(self):
        if self.simulation.advance():
            self.central.updateData()
        else:
            self.timer.stop()


def main(simulation: Simulation):
    """
    Launches Qt application for visualizing simulation.

    :param simulation:
    """
    import sys

    # TODO: Read simulation data from hdf5 file
    # TODO: MoviePy
    app = QtGui.QApplication(sys.argv)
    graphics = Graphics(simulation)
    graphics.show()
    sys.exit(app.exec_())
