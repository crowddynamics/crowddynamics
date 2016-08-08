from PyQt4 import QtGui, QtCore
import pyqtgraph as pg

from .graphics import SimulationGraphics
from .gui import Ui_MainWindow


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        """
        Overview
        --------
        Graphical user interface for Crowd Dynamics simulation using

        - PyQt4 [pyqt4]_
        - Pyqtgraph [pyqtgraph]_

        Design greatly inspired by [rtgraph]_.

        Main Window
        -----------
        Layout for the main window is created by using Qt designer.

        Graphics
        --------
        Graphics are implemented using pyqtgraph.

        Communication
        -------------
        Communication with simulation data.

        .. [pyqt4] Hess, D., & Summerfield, M. (2013). PyQt Whitepaper.

        .. [pyqtgraph] Campagnola, L. (2014). PyQtGraph - Scientific Graphics
           and GUI Library for Python. Posledn{’\i} Aktualizace. article.
           Retrieved from http://www.pyqtgraph.org/

        .. [rtgraph] Sepúlveda, S., Reyes, P., & Weinstein, A. (2015).
           Visualizing physiological signals in real-time.
           PROC. OF THE 14th PYTHON IN SCIENCE CONF.
           Retrieved from https://github.com/ssepulveda/RTGraph
        """
        super(MainWindow, self).__init__()
        # TODO: density, navigation visualization

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
        self.ui.plt.setBackground(background=None)
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

        self.ui.runSimulation.setCheckable(True)
        self.ui.runSimulation.setEnabled(False)
        self.ui.runSimulation.clicked.connect(self.run)

        self.ui.saveSimulation.setEnabled(False)
        self.ui.saveSimulation.clicked.connect(self.save)

    def init_simulation(self):
        # TODO: Importer
        from ..examples.evacuation import evacuation
        from ..examples.hallway import hallway
        from ..examples.outdoor import outdoor

        path = "/home/jaan/Dropbox/Projects/Crowd-Dynamics-Simulations/"
        name = self.ui.simulationName.currentText()
        kw = dict(
            size=self.ui.agentSize.value(),
            height=self.ui.heightBox.value(),
            width=self.ui.widthBox.value(),
            agent_model=self.ui.agentModel.currentText(),
            body_type=self.ui.bodyType.currentText(),
            path=path,
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

        # Enable controls
        self.ui.runSimulation.setEnabled(True)
        self.ui.saveSimulation.setEnabled(True)

    def plot_update(self):
        print("plot update")
        if self.simulation.advance():
            self.central.updateData()
        else:
            self.stop()

    def save(self):
        pass

    def start(self):
        print("Start")
        if not self.timer_plot_update.isActive():
            self.timer_plot_update.start(0)

    def stop(self):
        print("Stop")
        if self.timer_plot_update.isActive():
            self.timer_plot_update.stop()

    def run(self):
        print("Run")
        if self.simulation is None:
            raise Warning("Simulation is not initialized.")
        if self.ui.runSimulation.isChecked():
            self.start()
        else:
            self.stop()
