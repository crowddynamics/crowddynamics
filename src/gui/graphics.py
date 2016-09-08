import logging

import numpy as np
import pyqtgraph as pg
from shapely.geometry import LineString
from shapely.geometry import Polygon

from src.config import Load
from src.multiagent.simulation import MultiAgentSimulation


class Circular(pg.PlotDataItem):
    def __init__(self, radius):
        super(Circular, self).__init__()

        self.settings = Load().yaml("graphics")["agent"]
        self.radius = radius

        symbol_size = 2 * radius
        symbol_pen = np.zeros_like(radius, dtype=object)
        symbol_brush = np.zeros_like(radius, dtype=object)

        kwargs = dict(
            pxMode=False,
            pen=None,
            symbol='o',
            symbolSize=symbol_size,
            symbolPen=symbol_pen,
            symbolBrush=symbol_brush,
        )

        for key, val in self.settings["active"].items():
            kwargs[key][:] = val

        self.setData(**kwargs)

    def set_data(self, position, **kwargs):
        """
        =========== ==
        **Kwargs**

        *active*    --
        *strategy*  --
        =========== ==

        :param position: Positional data (x and y coordinates).
        :param kwargs:
        """
        active = kwargs.get("active", None)
        strategy = kwargs.get("strategy", None)

        if strategy is not None:
            # {0: "Impatient", 1: "Patient"}
            impatient = strategy == 0
            patient = strategy == 1

            for key, val in self.settings["impatient"].items():
                self.opts[key][impatient] = val

            for key, val in self.settings["patient"].items():
                self.opts[key][patient] = val

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

    def set_data(self, position, position_ls, position_rs, **kwargs):
        # TODO: orientation_indicator
        self.left_shoulder.set_data(position=position_ls, **kwargs)
        self.right_shoulder.set_data(position=position_rs, **kwargs)
        self.torso.set_data(position=position, **kwargs)


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
            if isinstance(domain, Polygon):
                x, y = domain.exterior.xy
                x, y = np.asarray(x), np.asarray(y)
                self.setRange(xRange=(x.min(), x.max()),
                              yRange=(y.min(), y.max()))
                item = pg.PlotDataItem(x, y)  # settings["domain"]["brush"]
                # self.addItem(item)

        if process.exits is not None:
            logging.debug("exits")
            exits = process.exits
            for exit_ in exits:
                if isinstance(exit_, LineString):
                    x, y = exit_.xy
                    x, y = np.asarray(x), np.asarray(y)
                    item = pg.PlotDataItem(x, y)  # TODO: pen
                    # self.addItem(item)

        if process.agent is not None:
            logging.debug("agent")
            agent = process.agent
            if agent.three_circle:
                model = ThreeCircle(agent.r_t, agent.r_s)
                model.set_data(agent.position, agent.position_ls,
                               agent.position_rs, active=agent.active)
                for item in model.items:
                    self.addItem(item)
            else:
                model = Circular(agent.radius)
                model.set_data(agent.position, active=agent.active)
                self.addItem(model)
            self.agent = model

        if process.obstacles is not None:
            logging.debug("Obstacles")
            obstacles = process.obstacles
            for obstacle in obstacles:
                if isinstance(obstacle, LineString):
                    x, y = obstacle.xy
                    x, y = np.asarray(x), np.asarray(y)
                    item = pg.PlotDataItem(x, y)
                    self.addItem(item)

    def update_data(self, data):
        """Update dynamic items."""
        for key, values in data.items():
            getattr(self, key).set_data(**values)
