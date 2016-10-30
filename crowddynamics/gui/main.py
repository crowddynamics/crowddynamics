import importlib
import logging
import sys
from functools import partial
from multiprocessing import Queue

import pyqtgraph as pg
from PyQt4 import QtGui, QtCore

from crowddynamics.functions import load_config
from .graphics import MultiAgentPlot
from .ui.gui import Ui_MainWindow

# Do not use multiprocessing in windows because of different semantics compared
# to linux.
enable_multiprocessing = not sys.platform.startswith('Windows')


class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    """
    Main window for the grahical user interface. Layout is created by using
    qtdesigner and the files can be found in the *designer* folder. Makefile
    to generate python code from the designer files can be used with command::

       make gui

    Main window consists of

    - Menubar (top)
    - Sidebar (left)
    - Graphics layout widget (middle)
    - Control bar (bottom)
    """

    def __init__(self):
        super(MainWindow, self).__init__()

        # Logger
        self.logger = logging.getLogger("crowddynamics.gui.mainwindow")

        # Load ui files
        self.setupUi(self)

        # Loading data from configs
        self.configs = load_config("simulations.yaml")

        # Simulation with multiprocessing
        self.queue = Queue(maxsize=4)
        self.process = None

        # Graphics
        self.timer = QtCore.QTimer(self)
        self.plot = None

        # Buttons
        # RadioButton for initializing HDF5 saving for the simulation
        self.savingButton = QtGui.QRadioButton("Save to HDF5Store")
        # Button that initializes selected simulation
        self.initButton = QtGui.QPushButton("Initialize Simulation")

        # Configures. Should be last.
        self.configure_plot()
        self.configure_signals()

    def enable_controls(self, boolean):
        self.startButton.setEnabled(boolean)
        self.stopButton.setEnabled(boolean)
        self.saveButton.setEnabled(boolean)

    def configure_plot(self):
        """Graphics widget for plotting simulation data."""
        self.logger.info("")
        pg.setConfigOptions(antialias=True)
        self.graphicsLayout.setBackground(None)
        self.plot = MultiAgentPlot()
        self.graphicsLayout.addItem(self.plot, 0, 0)

    def configure_signals(self):
        """Sets the functionality and values for the widgets."""
        self.logger.info("")

        # Buttons
        self.timer.timeout.connect(self.update_plots)
        self.startButton.clicked.connect(self.start)
        self.stopButton.clicked.connect(self.stop)
        self.initButton.clicked.connect(self.set_simulation)

        # Disable until simulation is set
        self.enable_controls(False)

        # Menus
        names = tuple(self.configs["simulations"].keys())
        self.simulationsBox.addItem("")  # No simulation. Clear sidebar.
        self.simulationsBox.addItems(names)
        self.simulationsBox.currentIndexChanged[str].connect(self.set_sidebar)

    def reset_buffers(self):
        self.logger.info("")
        while not self.queue.empty():
            self.queue.get()

    def clear_sidebar(self):
        # http://stackoverflow.com/questions/4528347/clear-all-widgets-in-a-layout-in-pyqt
        self.logger.info("")
        layout = self.sidebarLeft
        for i in reversed(range(layout.count())):
            if i in (0, 1):
                continue
            self.logger.debug("{}".format(layout.itemAt(i)))
            layout.itemAt(i).widget().setParent(None)

    def set_sidebar(self, name):
        self.logger.info(name)

        self.clear_sidebar()

        if name == "":
            return

        kwarg_mapping = self.configs["kwarg_mapping"]
        kwargs = self.configs["simulations"][name]["kwargs"]

        def _update(key, value):
            # FIXME
            self.logger.debug("Setting \"{}\" to \"{}\"".format(key, value))
            kwargs[key] = value

        for key, val in kwargs.items():
            self.logger.debug("{}: {}".format(key, val))
            # Set valid values and current value
            label = QtGui.QLabel(key)
            values = kwarg_mapping[key]
            update = partial(_update, key)

            if isinstance(val, int):
                widget = QtGui.QSpinBox()

                if values[0] is not None:
                    widget.setMinimum(values[0])
                else:
                    widget.setMinimum(-100000)

                if values[1] is not None:
                    widget.setMaximum(values[1])
                else:
                    widget.setMaximum(100000)

                widget.setValue(val)
                widget.valueChanged.connect(update)
            elif isinstance(val, float):
                widget = QtGui.QDoubleSpinBox()

                inf = float("inf")
                if values[0] is not None:
                    widget.setMinimum(values[0])
                else:
                    widget.setMinimum(-inf)

                if values[1] is not None:
                    widget.setMaximum(values[1])
                else:
                    widget.setMaximum(inf)

                widget.setValue(val)
                widget.valueChanged.connect(update)
            elif isinstance(val, bool):
                widget = QtGui.QRadioButton()
                widget.setChecked(val)
                widget.toggled.connect(update)
            elif isinstance(val, str):
                widget = QtGui.QComboBox()
                widget.addItems(values)
                index = widget.findText(val)
                widget.setCurrentIndex(index)
                widget.currentIndexChanged[str].connect(update)
            else:
                self.logger.warning(
                    "Value type not supported: {}".format(type(val)))

            self.sidebarLeft.addWidget(label)
            self.sidebarLeft.addWidget(widget)

        self.sidebarLeft.addWidget(self.savingButton)
        self.sidebarLeft.addWidget(self.initButton)

    def set_simulation(self):
        self.logger.info("")

        self.reset_buffers()

        # Import simulation from examples and initializes it.
        name = self.simulationsBox.currentText()

        d = self.configs["simulations"][name]
        module = importlib.import_module(d["module"])
        simulation = getattr(module, d["class"])
        self.process = simulation(self.queue, **d["kwargs"])

        # Enable controls
        self.enable_controls(True)

        # TODO: better format
        # Plot Simulation
        self.plot.configure(self.process)

        # Queing dictates what data is sent to graphics for display. For example
        # positions of agents.
        args = [(("agent", "agent"),
                 ["position", "active", "position_ls", "position_rs"])]

        if self.process.game is not None:
            args.append((("game", "agent"), ["strategy"]))

        self.process.configure_queuing(args)
        if self.savingButton.isChecked():
            self.process.configure_hdfstore()

    def update_plots(self):
        """Updates the data in the plot(s)."""
        data = self.queue.get()
        if data is None:
            self.timer.stop()
            self.enable_controls(False)
            self.process = None
            self.reset_buffers()
        else:
            if not enable_multiprocessing:
                self.process.update()  # Sequential processing
            self.plot.update_data(data)

    def start(self):
        """Start simulation process and updating plot."""
        self.startButton.setEnabled(False)
        if self.process is not None:
            self.logger.info("")

            if enable_multiprocessing:
                self.process.start()
            else:
                self.process.update()

            self.timer.start(0.01 * 1000)  # same as dt used in simulation
        else:
            self.logger.info("Process is not set")

    def stop(self):
        """Stops simulation process and updating the plot"""
        if self.process is not None:
            self.logger.info("")

            if enable_multiprocessing:
                self.process.stop()
            else:
                self.queue.put(None)
        else:
            self.logger.info("Process is not set")
