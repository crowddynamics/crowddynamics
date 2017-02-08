import logging
from timeit import default_timer as timer

import numpy as np
import pyqtgraph as pg
from shapely.geometry import LineString
from shapely.geometry import Polygon

from crowddynamics.functions import load_config
from crowddynamics.logging import log_with
from crowddynamics.multiagent.simulation import MultiAgentSimulation


class Circular(pg.PlotDataItem):
    def __init__(self, radius):
        super(Circular, self).__init__()

        self.settings = load_config("graphics.yaml")["agent"]
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
        else:
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
    r"""MultiAgentPlot is GraphicsItem for displaying simulation graphics.

    """
    logger = logging.getLogger(__name__)

    def __init__(self, parent=None):
        super(MultiAgentPlot, self).__init__(parent)

        # Plot settings
        self.setAspectLocked(lock=True, ratio=1)  # One to one scale
        self.showGrid(x=True, y=True, alpha=0.25)
        self.disableAutoRange()

        # Timer for fps
        self.last_time = timer()
        self.fps = None

        # Dynamics plot items
        self.agent = None

    @log_with(logger)
    def configure(self, process):
        r"""
        Configure static plot items and initial configuration of dynamic plot
        items (agents).

        Args:
            process (MultiAgentSimulation): Simulation process
        """
        # Clear previous plots and items
        self.clearPlots()
        self.clear()

        if process.domain is not None:
            self.logger.debug("domain")
            domain = process.domain
            if isinstance(domain, Polygon):
                x, y = domain.exterior.xy
                x, y = np.asarray(x), np.asarray(y)
                self.setRange(xRange=(x.min(), x.max()),
                              yRange=(y.min(), y.max()))
                item = pg.PlotDataItem(x, y)  # settings["domain"]["brush"]
                # self.addItem(item)

        if process.targets is not None:
            self.logger.debug("exits")
            exits = process.targets
            for exit_ in exits:
                if isinstance(exit_, LineString):
                    x, y = exit_.xy
                    x, y = np.asarray(x), np.asarray(y)
                    item = pg.PlotDataItem(x, y)  # TODO: pen
                    # self.addItem(item)

        if process.agent is not None:
            self.logger.debug("agent")
            agent = process.agent
            if agent.three_circle:
                model = ThreeCircle(agent.r_t, agent.r_s)
                # FIXME
                position, position_ls, position_rs = agent.positions(agent.indices())
                model.set_data(position, position_ls, position_rs, active=agent.active)
                for item in model.items:
                    self.addItem(item)
            else:
                model = Circular(agent.radius)
                model.set_data(agent.position, active=agent.active)
                self.addItem(model)
            self.agent = model

        if process.obstacles is not None:
            self.logger.debug("Obstacles")
            obstacles = process.obstacles
            for obstacle in obstacles:
                if isinstance(obstacle, LineString):
                    x, y = obstacle.xy
                    x, y = np.asarray(x), np.asarray(y)
                    item = pg.PlotDataItem(x, y)
                    self.addItem(item)

    def update_data(self, data):
        r"""
        Update dynamic items.

        Args:
            data:
        """
        for key, values in data.items():
            getattr(self, key).set_data(**values)

        # Frames per second
        now = timer()
        dt = now - self.last_time
        self.last_time = now
        fps = 1.0 / dt

        if self.fps is None:
            self.fps = fps
        else:
            s = np.clip(3 * dt, 0, 1)
            self.fps = self.fps * (1 - s) + fps * s
        self.setTitle('%0.2f fps' % self.fps)
