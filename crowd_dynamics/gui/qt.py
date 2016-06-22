import sys
from time import sleep

import numpy as np
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore

from crowd_dynamics.area import GoalRectangle
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
"""


def _examples():
    import pyqtgraph.examples as examples
    examples.run()


def gui(simulation: Simulation):
    # Qt application
    app = QtGui.QApplication(sys.argv)

    # Remote processing for graphics. Speeds up plotting.
    # import pyqtgraph.widgets.RemoteGraphicsView
    # view = pg.widgets.RemoteGraphicsView.RemoteGraphicsView()
    # view.pg.setConfigOptions(antialias=True)
    # rplt = view.pg.PlotItem()
    # rplt._setProxyOptions(deferGetattr=True)
    # view.setCentralItem(rplt)
    # rplt.plot(data, clear=True, _callSync='off')

    # Graphics window. Contains all the plots.
    win = pg.GraphicsWindow(title="Crowd Dynamics", size=(1200, 800))

    # Figure
    figure = win.addPlot(title="Simulation", row=0, col=0)
    pg.setConfigOptions(antialias=True)

    # One to one scale for x and y coordinates
    figure.setAspectLocked(lock=True, ratio=1)

    # PlotItems for plotting data
    areas = figure.plot()
    agent_ls = figure.plot()
    agent_rs = figure.plot()
    agent_c = figure.plot()
    direction = figure.plot()
    walls = figure.plot()

    # Agents. Plotted as circles
    # TODO: Orientable agent
    agent_ls.setData(symbol='o',
                     symbolSize=2 * simulation.agent.r_s,
                     connect=np.zeros(simulation.agent.size),
                     pxMode=False)
    agent_rs.setData(symbol='o',
                     symbolSize=2 * simulation.agent.r_s,
                     connect=np.zeros(simulation.agent.size),
                     pxMode=False)
    agent_c.setData(symbol='o',
                    symbolSize=2 * simulation.agent.r_t,
                    connect=np.zeros(simulation.agent.size),
                    pxMode=False)
    # Agent direction indicator
    connect = np.ones(3 * simulation.agent.size)
    connect[2::3] = np.zeros(simulation.agent.size)
    direction.setData(connect=connect)

    # Walls as lines or circles
    for wall in simulation.wall:
        if isinstance(wall, LinearWall):
            connect = np.zeros(2 * wall.size)
            connect[::2] = np.ones(wall.size)
            walls.setData(wall.params[:, :, 0].flatten(),
                          wall.params[:, :, 1].flatten(),
                          connect=connect)
        elif isinstance(wall, RoundWall):
            walls.setData(wall.params[:, :2],
                          symbol='o',
                          symbolSize=wall.params[:, 2],
                          connect=np.zeros(wall.size),
                          pxMode=False)
    # Areas as
    for area in simulation.goals:
        if isinstance(area, GoalRectangle):
            # areas.setData()
            pass

    timer = QtCore.QTimer()

    def update():
        if simulation.advance():
            agent_c.setData(simulation.agent.position)
            agent_ls.setData(simulation.agent.position_ls)
            agent_rs.setData(simulation.agent.position_rs)

            array = np.concatenate((simulation.agent.position_ls,
                                    simulation.agent.front,
                                    simulation.agent.position_rs),
                                   axis=1).reshape(
                3 * simulation.agent.shape[0],
                simulation.agent.shape[1])
            direction.setData(array)
        else:
            timer.stop()
            # exit()

    sleep(2.0)
    # Start and Stop buttons
    timer.timeout.connect(update)
    timer.start(0)

    # Start the Qt event loop
    sys.exit(app.exec_())
