import logging
import pyqtgraph as pg


class MultiAgentPlot(pg.PlotItem):
    def __init__(self, parent=None):
        """GraphicsItem for displaying simulation graphics."""
        super(MultiAgentPlot, self).__init__(parent)
        self.queue = None

        # Plot settings
        self.setAspectLocked(lock=True, ratio=1)  # One to one scale
        self.showGrid(x=True, y=True, alpha=0.25)
        self.disableAutoRange()

    def set(self, queue):
        logging.info("")
        self.queue = queue

    def update_plot(self):
        logging.debug("")
        data = self.queue.get()
