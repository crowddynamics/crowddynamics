import numpy as np
import numba
import logging
import pyqtgraph as pg

from src.core.vector2d import rotate270


@numba.jit(nopython=True, nogil=True)
def update_shoulder_positions(position, orientation, active, r_ts, r_t):
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
        self.radius = radius
        circle = dict(
            pen=None,
            symbol='o',
            symbolSize=2 * radius,
            symbolBrush=None,
            symbolPen=None,
            pxMode=False,
        )
        super(Circular, self).__init__(**circle)

    def setData(self, position, active):
        super(Circular, self).setData(position)


class ThreeCircle:
    def __init__(self, r_t, r_s, r_ts):
        self.left_shoulder = Circular(r_s)
        self.right_shoulder = Circular(r_s)
        self.torso = Circular(r_t)
        self.orientation = None

    def setData(self):
        pass


class FilledPolygon:
    def __init__(self):
        pass


class MultiAgentPlot(pg.PlotItem):
    def __init__(self, queue, parent=None):
        """GraphicsItem for displaying simulation graphics."""
        super(MultiAgentPlot, self).__init__(parent)
        self.queue = queue

        # Plot settings
        self.setAspectLocked(lock=True, ratio=1)  # One to one scale
        self.showGrid(x=True, y=True, alpha=0.25)
        self.disableAutoRange()

    def configure(self):
        logging.info("")

    def update_plot(self):
        logging.debug("")
        data = self.queue.get()
