import numpy as np
import numba
import logging
import pyqtgraph as pg

from src.config import Load
from src.core.vector2d import rotate270
from src.multiagent import MultiAgentSimulation
from src.structure import area
from src.structure.obstacle import LinearObstacle


@numba.jit(nopython=True, nogil=True)
def shoulder_positions(position, orientation, active, r_ts, r_t):
    indices = np.arange(len(position))[active]
    position_ls = np.zeros_like(position)
    position_rs = np.zeros_like(position)
    front = np.zeros_like(position)
    for i in indices:
        n = np.array((np.cos(orientation[i]), np.sin(orientation[i])))
        t = rotate270(n)
        offset = t * r_ts[i]
        position_ls[i] = position[i] - offset
        position_rs[i] = position[i] + offset
        front[i] = position[i] + n * r_t[i]
    return position_ls, position_rs, front


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

    def set_data(self, position, active):
        logging.debug("")
        for key, val in self.settings["active"].items():
            self.opts[key][active] = val

        for key, val in self.settings["inactive"].items():
            self.opts[key][active ^ True] = val

        self.setData(position)
        # self.updateItems()


class ThreeCircle:
    def __init__(self, r_t, r_s, r_ts):
        self.r_ts = r_ts

        # PlotItems
        self.left_shoulder = Circular(r_s)
        self.right_shoulder = Circular(r_s)
        self.torso = Circular(r_t)
        self.orientation_indicator = None
        self.items = (self.left_shoulder, self.right_shoulder, self.torso)

    def set_data(self, position, orientation, active):
        position_ls, position_rs, front = shoulder_positions(
            position, orientation, active, self.r_ts, self.torso.radius
        )
        self.left_shoulder.setData(position_ls, active)
        self.right_shoulder.setData(position_rs, active)
        self.torso.setData(position, active)
        # TODO: orientation_indicator


class Rectangle(pg.FillBetweenItem):
    def __init__(self, x, y, brush=None):
        c1 = pg.PlotDataItem(x, (y[0], y[0]))
        c2 = pg.PlotDataItem(x, (y[1], y[1]))
        super(Rectangle, self).__init__(c1, c2, brush=brush)


class MultiAgentPlot(pg.PlotItem):
    def __init__(self, queue, parent=None):
        """GraphicsItem for displaying simulation graphics."""
        super(MultiAgentPlot, self).__init__(parent)
        self.queue = queue

        # Plot settings
        self.setAspectLocked(lock=True, ratio=1)  # One to one scale
        self.showGrid(x=True, y=True, alpha=0.25)
        self.disableAutoRange()

        # Dynamics plot items
        self.dynamic_items = None

    def configure(self, process: MultiAgentSimulation):
        """Configure static plot items and initial configuration of dynamic
        plot items (agents).

        :param process: Simulation process
        :return:
        """
        logging.info("")
        self.clearPlots()

        if process.domain is not None:
            logging.debug("domain")
            domain = process.domain
            if isinstance(domain, area.Rectangle):
                self.setRange(xRange=domain.x, yRange=domain.y)
                self.addItem(Rectangle(domain.x, domain.y))

        if process.goals is not None:
            logging.debug("goals")
            goals = process.goals
            for goal in goals:
                if isinstance(goal, area.Rectangle):
                    self.addItem(Rectangle(goal.x, goal.y))

        if process.exits is not None:
            logging.debug("exits")
            pass

        if process.agent is not None:
            logging.debug("agent")
            agent = process.agent
            if agent.three_circle:
                model = ThreeCircle(agent.r_t, agent.r_s, agent.r_ts)
                model.set_data(agent.position, agent.angle, agent.active)
                for item in model.items:
                    self.addItem(item)
            else:
                model = Circular(agent.radius)
                model.set_data(agent.position, agent.active)
                self.addItem(model)

        if process.walls is not None:
            logging.debug("walls")
            walls = process.walls
            for wall in walls:
                if isinstance(wall, LinearObstacle):
                    connect = np.zeros(2 * wall.size, dtype=np.int32)
                    connect[::2] = np.ones(wall.size, dtype=np.int32)
                    walls.setData(wall.params[:, :, 0].flatten(),
                                  wall.params[:, :, 1].flatten(),
                                  connect=connect)

    def update_data(self):
        logging.debug("")
        data = self.queue.get()
