from PyQt4 import QtGui, QtCore
import pyqtgraph as pg

from .gui import Ui_MainWindow

from src.Qt.graphics import SimulationGraphics
from src.examples.evacuation import evacuation
from src.examples.hallway import hallway
from src.examples.outdoor import outdoor


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Load ui files
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Graphics
        self.central = None
        self.simulation = None
        self.timer_plot_update = None

        # Configures
        self.configure_plot()
        self.configure_timers()
        self.configure_signals()

    def configure_plot(self):
        pg.setConfigOptions(antialias=True)
        # self.ui.plt.setBackground(background=None)
        self.ui.plt.setAntialiasing(True)
        self.central = SimulationGraphics()
        self.ui.plt.addItem(self.central, 0, 0)

    def configure_timers(self):
        self.timer_plot_update = QtCore.QTimer(self)
        self.timer_plot_update.timeout.connect(self.plot_update)

    def configure_signals(self):
        # Default values for the fields
        self.ui.agentSize.setValue(30)
        self.ui.heightBox.setValue(10.0)
        self.ui.widthBox.setValue(10.0)

        # Select simulation
        # self.ui.simulationName.currentIndexChanged[str].connect()

        # Simulation controls
        self.ui.initSimulation.clicked.connect(self.init_simulation)

        self.ui.runSimulation.setEnabled(False)
        self.ui.runSimulation.setCheckable(True)
        self.ui.runSimulation.clicked.connect(self.run)

        self.ui.saveSimulation.setEnabled(False)
        self.ui.saveSimulation.clicked.connect(self.save)

    def init_simulation(self):
        name = self.ui.simulationName.currentText()
        kw = dict(
            size=self.ui.agentSize.value(),
            height=self.ui.heightBox.value(),
            width=self.ui.widthBox.value(),
            agent_model=self.ui.agentModel.currentText(),
            body_type=self.ui.bodyType.currentText()
        )

        if name == "evacuation":
            self.simulation = evacuation(**kw)
        elif name == "hallway":
            self.simulation = hallway(**kw)
        elif name == "outdoor":
            self.simulation = outdoor(**kw)
        else:
            self.simulation = None
            self.ui.runSimulation.setEnabled(False)
            self.ui.saveSimulation.setEnabled(False)
            return

        self.central.setSimulation(self.simulation)
        self.ui.runSimulation.setEnabled(True)
        self.ui.saveSimulation.setEnabled(True)

    def plot_update(self):
        self.central.updateData()

    def save(self):
        pass

    def start(self):
        if not self.timer_plot_update.isActive() and \
                        self.simulation is not None:
            self.timer_plot_update.start(0)

    def stop(self):
        if self.timer_plot_update.isActive():
            self.timer_plot_update.stop()

    def run(self):
        if self.ui.runSimulation.isChecked():
            self.start()
        else:
            self.stop()
