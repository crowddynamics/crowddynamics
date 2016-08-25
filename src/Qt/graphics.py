import logging

import numpy as np
import pyqtgraph as pg

from src.geometry.curve import LinearObstacle
from src.config import Load
from src.geometry import surface
from src.multiagent.simulation import MultiAgentSimulation


class Circular(pg.PlotDataItem):
    def __init__(self, radius):
        super(Circular, self).__init__()
        self.settings = Load().yaml("graphics")["agent"]
        self.radius = radius

        symbol_size = 2 * radius
        symbol_pen = np.zeros_like(radius, dtype=object)
        symbol_brush = np.zeros_like(radius, dtype=object)

        kwargs = dict(pxMode=False,
                      pen=None,
                      symbol='o',
                      symbolSize=symbol_size,
                      symbolPen=symbol_pen,
                      symbolBrush=symbol_brush, )

        for key, val in self.settings["active"].items():
            logging.debug("{}: {}".format(key, val))
            kwargs[key][:] = val

        logging.debug("{}".format(kwargs))

        self.setData(**kwargs)

    def set_data(self, position, active, **kwargs):
        for key, val in self.settings["active"].items():
            self.opts[key][active] = val

        for key, val in self.settings["inactive"].items():
            self.opts[key][active ^ True] = val

        self.setData(position)


class ThreeCircle:
    def __init__(self, r_t, r_s):
        # PlotItems
        self.left_shoulder = Circular(r_s)
        self.right_shoulder = Circular(r_s)
        self.torso = Circular(r_t)
        self.orientation = None
        self.items = (self.left_shoulder, self.right_shoulder, self.torso)

    def set_data(self, position, position_ls, position_rs, active, **kwargs):
        # TODO: orientation_indicator
        self.left_shoulder.set_data(position_ls, active)
        self.right_shoulder.set_data(position_rs, active)
        self.torso.set_data(position, active)


class Rectangle(pg.FillBetweenItem):
    def __init__(self, x, y, brush=None):
        c1 = pg.PlotDataItem(x, (y[0], y[0]))
        c2 = pg.PlotDataItem(x, (y[1], y[1]))
        super(Rectangle, self).__init__(c1, c2, brush=brush)


class MultiAgentPlot(pg.PlotItem):
    def __init__(self, parent=None):
        """GraphicsItem for displaying simulation graphics."""
        super(MultiAgentPlot, self).__init__(parent)

        # Plot settings
        self.setAspectLocked(lock=True, ratio=1)  # One to one scale
        self.showGrid(x=True, y=True, alpha=0.25)
        self.disableAutoRange()

        # Dynamics plot items
        self.agent = None

    def configure(self, process: MultiAgentSimulation):
        """Configure static plot items and initial configuration of dynamic
        plot items (agents).

        :param process: Simulation process
        :return:
        """
        logging.info("")

        # Clear previous plots and items
        self.clearPlots()
        self.clear()

        # Setup plots
        settings = Load().yaml("graphics")

        if process.domain is not None:
            logging.debug("domain")
            domain = process.domain
            if isinstance(domain, surface.Rectangle):
                self.setRange(xRange=domain.x, yRange=domain.y)
                self.addItem(Rectangle(domain.x, domain.y, settings["domain"]["brush"]))

        if process.goals is not None:
            logging.debug("goals")
            goals = process.goals
            for goal in goals:
                if isinstance(goal, surface.Rectangle):
                    self.addItem(Rectangle(goal.x, goal.y, settings["goal"]["brush"]))

        if process.exits is not None:
            logging.debug("exits")
            pass

        if process.agent is not None:
            logging.debug("agent")
            agent = process.agent
            if agent.three_circle:
                model = ThreeCircle(agent.r_t, agent.r_s)
                model.set_data(agent.position, agent.position_ls,
                               agent.position_rs, agent.active)
                for item in model.items:
                    self.addItem(item)
            else:
                model = Circular(agent.radius)
                model.set_data(agent.position, agent.active)
                self.addItem(model)
            self.agent = model

        if process.walls is not None:
            logging.debug("walls")
            walls = process.walls
            for wall in walls:
                if isinstance(wall, LinearObstacle):
                    connect = np.zeros(2 * wall.size, dtype=np.int32)
                    connect[::2] = np.ones(wall.size, dtype=np.int32)
                    self.plot(wall.params[:, :, 0].flatten(),
                              wall.params[:, :, 1].flatten(),
                              connect=connect)

    def update_data(self, data):
        """Update dynamic items."""
        # logging.debug("")
        for key, values in data.items():
            getattr(self, key).set_data(**values)
