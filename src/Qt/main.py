from PyQt4 import QtGui, QtCore
import pyqtgraph as pg

import logging as log

from .graphics import SimulationPlot
from .gui import Ui_MainWindow


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Load ui files
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Graphics
        self.simulation_plot = None
        self.simulation = None
        self.timer_plot_update = None

        # Configures
        self.configure_plot()
        self.configure_timers()
        self.configure_signals()

    def configure_plot(self):
        log.info("Configuring graphics")
        pg.setConfigOptions(antialias=True)
        self.ui.plt.setBackground(background=None)
        self.simulation_plot = SimulationPlot()
        self.ui.plt.addItem(self.simulation_plot, 0, 0)

    def configure_timers(self):
        log.info("Configuring timers")
        self.timer_plot_update = QtCore.QTimer(self)
        self.timer_plot_update.timeout.connect(self.plot_update)

    def configure_signals(self):
        log.info("Configuring signals")
        # Default values for the fields
        self.ui.agentSize.setValue(30)
        self.ui.heightBox.setValue(10.0)
        self.ui.widthBox.setValue(10.0)

        # Select simulation
        # self.ui.simulationName.currentIndexChanged[str].connect()

        # Simulation controls
        self.ui.initSimulation.clicked.connect(self.new_simulation)

        self.ui.runSimulation.setCheckable(True)
        self.ui.runSimulation.setEnabled(False)
        self.ui.runSimulation.clicked.connect(self.run)

        self.ui.saveSimulation.setEnabled(False)
        self.ui.saveSimulation.clicked.connect(self.save)

    def new_simulation(self):
        # TODO: Importer
        from ..examples.evacuation import RoomEvacuation, \
            RoomEvacuationWithEgressGame
        from ..examples.hallway import Hallway
        from ..examples.outdoor import Outdoor

        # path = "/home/jaan/Dropbox/Projects/CrowdDynamicsSimulations/"

        name = self.ui.simulationName.currentText()
        kw = dict(
            size=self.ui.agentSize.value(),
            height=self.ui.heightBox.value(),
            width=self.ui.widthBox.value(),
            model=self.ui.agentModel.currentText(),
            body=self.ui.bodyType.currentText(),
        )

        if name == "evacuation":
            self.simulation = RoomEvacuation(**kw)
        elif name == "evacuation_game":
            self.simulation = RoomEvacuationWithEgressGame(**kw)
        elif name == "hallway":
            self.simulation = Hallway(**kw)
        elif name == "outdoor":
            self.simulation = Outdoor(**kw)
        else:
            self.simulation = None
            self.ui.runSimulation.setEnabled(False)
            self.ui.saveSimulation.setEnabled(False)
            return

        self.simulation_plot.set_simulation(self.simulation)

        log.info("Initializing simulation\n"
                 "Name: {}\n"
                 "Args: {}".format(name, kw))

        # Enable controls
        self.ui.runSimulation.setEnabled(True)
        self.ui.saveSimulation.setEnabled(True)

    def plot_update(self):
        # TODO: simulation stats
        if self.simulation.update():
            self.simulation_plot.update_data()
        else:
            self.stop()

    def save(self):
        if self.simulation is not None:
            log.info("Simulation saved")
            self.simulation.save()
        else:
            log.warning("Attempting to save but simulation is not initialized.")

    def start(self):
        if not self.timer_plot_update.isActive():
            log.info("Starting")
            self.timer_plot_update.start(0)

    def stop(self):
        if self.timer_plot_update.isActive():
            log.info("Stopping")
            self.timer_plot_update.stop()

    def run(self):
        if self.simulation is None:
            log.warning("Simulation is not initialized.")
            return

        if self.ui.runSimulation.isChecked():
            self.start()
        else:
            self.stop()
