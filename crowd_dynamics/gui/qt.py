from PyQt4 import QtGui
import pyqtgraph as pg

from crowd_dynamics.parameters import Parameters
from crowd_dynamics.struct.agent import Agent


class Gui:
    """
    arrowItem
    scatterPlotItem
    PlotDataItem
    http://zetcode.com/gui/pyqt4/
    http://www.pyqtgraph.org/documentation/plotting.html
    """
    def __init__(self):
        pass


def circles(pos, radius):
    # item = pg.PlotDataItem()
    item = pg.ScatterPlotItem()
    item.setData(pos=pos, size=radius, symbol='o', pxMode=False)
    return item

p = Parameters(100, 100)
agent = Agent(*p.agent(100))


# Qt application
# app = QtGui.QApplication([])
# top_widget = QtGui.QWidget()
# layout = QtGui.QGridLayout()
#
# top_widget.setLayout(layout)

# plot_widget = pg.PlotWidget()
plot_widget = pg.plot()

# layout.addWidget(plot_widget)

ditem = circles(agent.position, agent.radius)
plot_widget.plot(ditem)

# Display the widget as a new window
# top_widget.show()

# Start the Qt event loop
# app.exec_()
