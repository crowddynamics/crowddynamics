import logging as log
import os

import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt4 import QtGui, QtCore

from .graphics import SimulationPlot
from .game import Ui_Game
from .gui import Ui_MainWindow


class GameControls(QtGui.QWidget):
    def __init__(self, parent=None):
        super(GameControls, self).__init__(parent)
        self.parent = parent
        self.ui = Ui_Game()
        self.ui.setupUi(self)

        self.timer = None

        self.configure_timers()
        self.configure_signals()

    def configure_signals(self):
        self.ui.gameButton.setCheckable(True)
        self.ui.gameButton.clicked.connect(self.run)

    def configure_timers(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)

    def update_plot(self):
        t_aset = self.ui.tasetBox.value()
        self.parent.simulation.game.update(0, 1, t_aset)
        self.parent.simulation_plot.update_data()

    def start(self):
        if not self.timer.isActive():
            log.info("Starting")
            self.timer.start(0)

    def stop(self):
        if self.timer.isActive():
            self.timer.stop()

    def run(self):
        if self.ui.gameButton.isChecked():
            self.start()
        else:
            self.stop()


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Load ui files
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Graphics
        self.simulation_plot = None
        self.image_exporter = None
        self.simulation = None
        self.timer = None
        self.dirpath = None

        # Configures
        self.configure_plot()
        self.configure_timers()
        self.configure_signals()

        # Widgets. (Non hard coded widgets)
        self.widgets = []

    def get_directory(self):
        dlg = QtGui.QFileDialog()
        dlg.setFileMode(QtGui.QFileDialog.Directory)

    def configure_plot(self):
        log.info("Configuring graphics")
        pg.setConfigOptions(antialias=True)
        self.ui.plt.setBackground(background=None)
        self.simulation_plot = SimulationPlot()
        self.ui.plt.addItem(self.simulation_plot, 0, 0)

    def configure_timers(self):
        log.info("Configuring timers")
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)

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

        # QFileDialog for opening and saving simulation
        # Select dirpath for saving simulation
        self.ui.dirpathLine.setEnabled(False)
        self.ui.dirpathLine.setText(os.path.abspath("."))
        self.ui.saveButton.clicked.connect(self.save_button)

    def save_button(self):
        if self.ui.saveButton.isChecked():
            self.ui.dirpathLine.setEnabled(True)
        else:
            self.ui.dirpathLine.setEnabled(False)

    def new_simulation(self):
        # TODO: Importer
        from ..examples.evacuation import RoomEvacuation, RoomEvacuationGame
        from ..examples.hallway import Hallway
        from ..examples.outdoor import Outdoor

        for widget in self.widgets:
            self.ui.verticalLayout.removeWidget(widget)

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
            self.simulation = RoomEvacuationGame(**kw)
            widget = GameControls(self)
            self.ui.verticalLayout.addWidget(widget)
            self.widgets.append(widget)
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
        self.image_exporter = pg.exporters.ImageExporter(self.simulation_plot)

        self.dirpath = self.ui.dirpathLine.text()
        if self.ui.saveButton.isChecked():
            self.simulation.configure_saving(self.dirpath)

        log.info("Initializing simulation\n"
                 "Name: {}\n"
                 "Args: {}".format(name, kw))

        # Enable controls
        self.ui.runSimulation.setEnabled(True)
        self.ui.saveSimulation.setEnabled(True)

    def update_plot(self):
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
        if not self.timer.isActive():
            log.info("Starting")
            self.timer.start(0)

    def stop(self):
        if self.timer.isActive():
            log.info("Stopping")
            self.timer.stop()

    def run(self):
        if self.simulation is None:
            log.warning("Simulation is not initialized.")
            return

        if self.ui.runSimulation.isChecked():
            self.start()
        else:
            self.stop()
