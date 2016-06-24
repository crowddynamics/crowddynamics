import sys

import numpy as np
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore

from crowd_dynamics.area import GoalRectangle, Bounds
from crowd_dynamics.simulation import Simulation
from crowd_dynamics.structure.wall import LinearWall
from crowd_dynamics.structure.wall import RoundWall

"""
http://zetcode.com/gui/pyqt4/
http://www.pyqtgraph.org/documentation/plotting.html
http://stackoverflow.com/questions/24197910/live-data-monitor-pyqtgraph
http://stackoverflow.com/questions/18080170/what-is-the-easiest-way-to-achieve-realtime-plotting-in-pyqtgraph
http://www.pyqtgraph.org/documentation/graphicsItems/plotdataitem.html?highlight=plotdataitem#pyqtgraph.PlotDataItem
http://www.pyqtgraph.org/documentation/graphicsItems/viewbox.html

Remote processing for graphics. Speeds up plotting.
import pyqtgraph.widgets.RemoteGraphicsView
view = pg.widgets.RemoteGraphicsView.RemoteGraphicsView()
view.pg.setConfigOptions(antialias=True)
rplt = view.pg.PlotItem()
rplt._setProxyOptions(deferGetattr=True)
view.setCentralItem(rplt)
rplt.plot(data, clear=True, _callSync='off')

"""


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
        connect = np.ones(3 * self.agent.size)
        connect[2::3] = np.zeros(self.agent.size)
        self.left_shoulder = self.addCircle(self.agent.r_s)
        self.right_shoulder = self.addCircle(self.agent.r_s)
        self.center = self.addCircle(self.agent.r_t)
        self.direction = self.plot(connect=connect)

        # Walls
        self.walls = self.plot()
        self.addWalls()

    def addCircle(self, sizes):
        return self.plot(symbol='o', symbolSize=2 * sizes, pen=None, pxMode=False)

    def addWalls(self):
        for wall in self.simulation.wall:
            if isinstance(wall, LinearWall):
                connect = np.zeros(2 * wall.size)
                connect[::2] = np.ones(wall.size)
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


class Controls:
    def __init__(self):
        pass


def main(simulation: Simulation):
    # Qt application
    app = QtGui.QApplication(sys.argv)

    pg.setConfigOptions(antialias=True)

    # Graphics. Contains all the plots.
    layout = pg.GraphicsLayoutWidget()
    layout.setWindowTitle("Crowd Dynamics")
    layout.resize(*(1200, 800))
    layout.show()

    central = CentralItem(simulation)
    layout.addItem(central, 0, 0, 1, 1)  # row, col, rowspan, colspan

    timer = QtCore.QTimer()

    def update():
        if simulation.advance():
            central.updateData()
        else:
            timer.stop()

    timer.timeout.connect(update)
    timer.start(0)

    # Start the Qt event loop
    sys.exit(app.exec_())
