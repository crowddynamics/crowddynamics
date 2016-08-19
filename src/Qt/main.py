import importlib
from multiprocessing import Queue

import pyqtgraph as pg
from PyQt4 import QtGui, QtCore

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
        self.configs = self.configs_loader.yaml("simulations", ordered=True)

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
        self.simulationsBox.addItem("")  # No simulation. Clear sidebar.
        self.simulationsBox.addItems(tuple(names))
        self.simulationsBox.currentIndexChanged[str].connect(self.set_sidebar)

    def reset_buffers(self):
        while not self.queue.empty():
            self.queue.get()

    def clear_sidebar(self):
        # FIXME
        for widget in self.sidebarWidgets:
            self.sidebarLeft.removeWidget(widget)

    def set_sidebar(self, name):
        self.clear_sidebar()

        if name == "":
            return

        mapping = self.configs["kwarg_mapping"]
        configs = self.configs["simulations"][name]
        kwargs = configs["kwargs"]

        # TODO: Connect to dictionary
        for key, val in kwargs.items():
            # Set valid values and current value
            label = QtGui.QLabel(key)
            values = mapping[key]
            if isinstance(val, int):
                widget = QtGui.QSpinBox()
                widget.setMinimum(values[0])
                widget.setMaximum(values[1])
                widget.setValue(val)
            elif isinstance(val, float):
                widget = QtGui.QDoubleSpinBox()
                widget.setMinimum(values[0])
                widget.setMaximum(values[1])
                widget.setValue(val)
            elif isinstance(val, bool):
                widget = QtGui.QRadioButton()
                # widget.setChecked(val)
            elif isinstance(val, str):
                widget = QtGui.QComboBox()
                widget.addItems(values)
                index = widget.findText(val)
                widget.setCurrentIndex(index)
            else:
                continue

            self.sidebarWidgets.append(widget)
            self.sidebarLeft.addWidget(label)
            self.sidebarLeft.addWidget(widget)

        initButton = QtGui.QPushButton("Initialize")
        initButton.clicked.connect(self.set_simulation)
        self.sidebarLeft.addWidget(initButton)
        # self.sidebarLeft.addWidget(QtGui.QSpacerItem())

    def set_simulation(self):
        self.reset_buffers()
        name = self.simulationsBox.currentText()
        simu_dict = self.configs["simulations"][name]
        module_name = simu_dict["module"]
        class_name = simu_dict["class"]
        module = importlib.import_module(module_name)
        simulation = getattr(module, class_name)
        self.process = simulation(self.queue, **simu_dict["kwargs"])

    def update_plot(self):
        """Updates the data in the plot."""
        pass

    def start(self):
        """Start simulation process and updating plot."""
        if self.process is not None:
            self.process.start()
            self.timer.start(0)

    def stop(self):
        """Stops simulation process and updating the plot"""
        if self.process is not None:
            self.timer.stop()
            self.process.stop()
            self.process.join()
            self.reset_buffers()
