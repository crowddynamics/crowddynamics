import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
from multiprocessing import Queue

from src.configs.load import Load
from .graphics import MultiAgentPlot
from .ui.gui import Ui_MainWindow


class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Load ui files
        self.setupUi(self)

        # Loading data from configs
        self.configs_loader = Load()
        self.configs = self.configs_loader.yaml("simulations")

        # Simulation with multiprocessing
        self.queue = Queue()
        self.process = None

        # Graphics
        pg.setConfigOptions(antialias=True)
        self.plot = None

        self.timer = QtCore.QTimer(self)
        self.dirpath = None

        # Configures
        self.configure_plot()
        self.configure_timers()
        self.configure_signals()

        # Programatically laid widgets
        self.sidebarWidgets = []

    def configure_plot(self):
        """Graphics widget for plotting simulation data."""
        self.graphicsLayout.setBackground(background=None)
        self.plot = MultiAgentPlot()
        self.graphicsLayout.addItem(self.plot, 0, 0)

    def configure_signals(self):
        """Sets the functionality and values for the widgets."""
        # Buttons
        self.timer.timeout.connect(self.update_plot)
        self.startButton.clicked.connect(self.start)
        self.stopButton.clicked.connect(self.stop)
        # Menus
        names = self.configs["simulations"].keys()
        self.simulationsBox.addItems(names)
        self.simulationsBox.currentIndexChanged.connect(self.set_sidebar)

    def reset_buffers(self):
        while not self.queue.empty():
            self.queue.get()

    def set_sidebar(self, name):
        # self.configs
        # for widget in widgets:
        #     self.sidebarLeft.addWidget(widget)
        pass

    def clear_sidebar(self):
        for widget in self.sidebarWidgets:
            self.sidebarLeft.removeWidget(widget)

    def set_simulation(self):
        self.reset_buffers()
        self.clear_sidebar()
        # self.set_sidebar()

    def update_plot(self):
        """Updates the data in the plot."""
        pass

    def start(self):
        """Start simulation process and updating plot."""
        pass

    def stop(self):
        """Stops simulation process and updating the plot"""
        pass
