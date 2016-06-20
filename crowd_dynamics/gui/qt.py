import sys

import numpy as np
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore

from crowd_dynamics.area import GoalRectangle
from crowd_dynamics.simulation import Simulation
from crowd_dynamics.struct.wall import LinearWall
from crowd_dynamics.struct.wall import RoundWall

"""
arrowItem
scatterPlotItem
PlotDataItem
http://zetcode.com/gui/pyqt4/
http://www.pyqtgraph.org/documentation/plotting.html
http://stackoverflow.com/questions/24197910/live-data-monitor-pyqtgraph
http://stackoverflow.com/questions/18080170/what-is-the-easiest-way-to-achieve-realtime-plotting-in-pyqtgraph
http://www.pyqtgraph.org/documentation/graphicsItems/plotdataitem.html?highlight=plotdataitem#pyqtgraph.PlotDataItem
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

    # Figure
    win = pg.GraphicsWindow(title="Crowd Dynamics", size=(800, 800))
    figure = win.addPlot(title="Simulation")
    pg.setConfigOptions(antialias=True)

    agent_c = figure.plot()
    agent_ls = figure.plot()
    agent_rs = figure.plot()
    walls = figure.plot()
    areas = figure.plot()

    agent_c.setData(symbol='o', symbolSize=simulation.agent.r_t,
                    connect=np.zeros(simulation.agent.size), pxMode=False)

    agent_ls.setData(symbol='o', symbolSize=simulation.agent.r_s,
                     connect=np.zeros(simulation.agent.size), pxMode=False)

    agent_rs.setData(symbol='o', symbolSize=simulation.agent.r_s,
                     connect=np.zeros(simulation.agent.size), pxMode=False)

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

    for area in simulation.goals:
        if isinstance(area, GoalRectangle):
            # areas.setData()
            pass

    def update():
        if simulation.advance():
            agent_c.setData(simulation.agent.position)
            agent_ls.setData(simulation.agent.position_ls)
            agent_rs.setData(simulation.agent.position_rs)
        else:
            exit()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0)

    # Start the Qt event loop
    sys.exit(app.exec_())
