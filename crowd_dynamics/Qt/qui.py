import numpy as np
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore

from crowd_dynamics.simulation import Simulation
from crowd_dynamics.structure.area import Rectangle
from crowd_dynamics.structure.wall import LinearWall
from crowd_dynamics.structure.wall import RoundWall


class CentralItem(pg.PlotItem):
    name = "central_item"
    title = "Simulation"

    def __init__(self, simulation: Simulation):
        """Central plot."""
        # TODO: Remote processing
        # TODO: Legend
        # TODO: Coloring of agents (Forces, etc)
        super(CentralItem, self).__init__(name=self.name)

        # Data
        self.simulation = simulation
        agent = self.simulation.agent
        bounds = self.simulation.bounds

        # One to one scale for x and y coordinates
        self.setAspectLocked(lock=True, ratio=1)
        self.showGrid(True, True, 0.5)
        self.setLabels(title=self.title, left="y", bottom="x")
        if bounds is not None:
            if isinstance(bounds, Rectangle):
                self.setRange(xRange=bounds.x, yRange=bounds.y)
                self.disableAutoRange()

        # Areas
        if bounds is not None:
            if isinstance(bounds, Rectangle):
                # brush = pg.mkBrush(255, 255, 255, 255 // 4)  # White, transparent
                # c1 = pg.PlotDataItem([bounds.x[0]], [bounds.y[0]])
                # c2 = pg.PlotDataItem([bounds.x[1]], [bounds.y[1]])
                # pg.FillBetweenItem(c1, c2, brush=brush)
                pass
        # TODO: Goals

        # Agent
        brush_psy = pg.mkBrush(255, 255, 255, 255 // 4)  # White, transparent
        self.psy = self.addCircle(agent.radius,
                                  symbolPen=None,
                                  symbolBrush=brush_psy)

        connect = np.ones(3 * agent.size, dtype=np.int32)
        connect[2::3] = np.zeros(agent.size, dtype=np.int32)
        self.left_shoulder = self.addCircle(agent.r_s)
        self.right_shoulder = self.addCircle(agent.r_s)
        self.torso = self.addCircle(agent.r_t)
        self.direction = self.plot(connect=connect)

        # Walls
        self.walls = self.plot()
        self.addWalls()

        self.updateData()

    def addCircle(self, radius, **kwargs):
        return self.plot(symbol='o', symbolSize=2 * radius, pen=None,
                         pxMode=False, **kwargs)

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

    def updateData(self):
        """Updates data in the plot."""
        agent = self.simulation.agent

        brush = pg.mkBrush(0, 0, 255, 255)
        if self.simulation.egress_model is not None:
            impatient = pg.mkBrush(255, 0, 0, 255)  # RGBA
            patient = pg.mkBrush(0, 0, 255, 255)  # RGBA
            states = np.array((impatient, patient))
            brush = states[self.simulation.egress_model.strategy]

        self.psy.setData(agent.position)
        self.torso.setData(agent.position, symbolBrush=brush)
        self.left_shoulder.setData(agent.position_ls, symbolBrush=brush)
        self.right_shoulder.setData(agent.position_rs, symbolBrush=brush)

        array = np.concatenate((agent.position_ls,
                                agent.front,
                                agent.position_rs), axis=1)
        array = array.reshape(3 * agent.shape[0], agent.shape[1])
        self.direction.setData(array)

        text = "Iterations: {} " \
               "Simulation time: {:0.2f} " \
               "Agents in goal: {}"
        stats = self.simulation.result
        self.setLabels(top=text.format(stats.iterations, stats.simulation_time,
                                       stats.in_goal))


class Controls(QtGui.QWidget):
    def __init__(self):
        super().__init__()


class Monitor(pg.PlotItem):
    name = "monitor"
    title = "Monitor"

    def __init__(self, simulation: Simulation):
        """
        Plot visualizing simulation data.
        Egress times
        Forces
        Timestep
        """
        super(Monitor, self).__init__(name=self.name, title=self.title)
        self.showGrid(True, True, 0.5)
        self.simulation = simulation

    def updateData(self):
        pass


class Graphics(pg.GraphicsLayoutWidget):
    def __init__(self, simulation: Simulation, parent=None, **kargs):
        """Contains all the plots. Updates interactive plots."""
        super().__init__(parent, **kargs)

        self.simulation = simulation

        pg.setConfigOptions(antialias=True)
        self.setWindowTitle("Crowd Dynamics")
        self.resize(*(1200, 800))

        self.central = CentralItem(self.simulation)
        self.addItem(self.central, 0, 0, 1, 1)  # row, col, rowspan, colspan
        # self.monitor = Monitor(self.simulation)
        # self.addItem(self.monitor, 1, 0, 1, 1)  # row, col, rowspan, colspan

        self.timer = QtCore.QTimer()
        # noinspection PyUnresolvedReferences
        self.timer.timeout.connect(self.updatePlots)
        self.timer.start(0)

    def updatePlots(self):
        if self.simulation.advance():
            self.central.updateData()
            # self.monitor.updateData()
        else:
            self.timer.stop()


def main(simulation: Simulation):
    """Launches Qt application for visualizing simulation.
    :param simulation:
    """
    import sys

    # TODO: Read simulation data from hdf5 file
    # TODO: MoviePy

    app = QtGui.QApplication(sys.argv)
    graphics = Graphics(simulation)
    graphics.show()
    sys.exit(app.exec_())
