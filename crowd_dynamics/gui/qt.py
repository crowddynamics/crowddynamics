import sys

import numpy as np

from PyQt4 import QtGui, QtCore
import pyqtgraph as pg

from crowd_dynamics.parameters import Parameters

"""
arrowItem
scatterPlotItem
PlotDataItem
http://zetcode.com/gui/pyqt4/
http://www.pyqtgraph.org/documentation/plotting.html
http://stackoverflow.com/questions/24197910/live-data-monitor-pyqtgraph
http://www.pyqtgraph.org/documentation/graphicsItems/plotdataitem.html?highlight=plotdataitem#pyqtgraph.PlotDataItem
"""


def main():
    # Qt application
    app = QtGui.QApplication(sys.argv)

    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)

    # Plot
    win = pg.GraphicsWindow()
    win.resize(800, 800)
    win.setWindowTitle("Crowd Dynamics")
    field = win.addPlot(title="Agents")
    plot = field.plot()

    # widget = pg.ScatterPlotWidget()
    p = Parameters(100, 100)
    size = 300
    radius = np.random.uniform(0, 1.0, size=size)
    plot.setData(symbolSize=radius,
                 symbol='o',
                 pxMode=False,
                 connect=np.zeros(size),
                 name="name")

    def update():
        position = p.random_2d_coordinates(size)
        plot.setData(position)

    # update()
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(16)
    # Start the Qt event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
