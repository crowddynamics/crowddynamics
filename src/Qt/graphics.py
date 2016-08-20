import logging
import pyqtgraph as pg


class MultiAgentPlot(pg.PlotItem):
    def __init__(self, parent=None):
        """GraphicsItem for displaying simulation graphics."""
        super(MultiAgentPlot, self).__init__(parent)
        self.queue = None

    def set(self, queue):
        logging.info("")
        self.queue = queue

    def update_plot(self):
        logging.debug("")
        data = self.queue.get()
