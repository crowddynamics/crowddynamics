from collections import OrderedDict

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QComboBox, QCheckBox,  QSpinBox, QDoubleSpinBox

import pyqtgraph as pg
from src.Qt.graphics import SimulationGraphics
from src.examples.evacuation import evacuation
from src.examples.hallway import hallway
from src.examples.outdoor import outdoor


kw = ("size", "width", "height", "agent_model", "body_type")

kw = dict(
    outdoor=kw,
    evacuation=kw + ("door_width",
                     "exit_hall_width",
                     "spawn_shape",
                     "egress_model",
                     "t_aset"),
    hallway=kw
)


class Controls(QtGui.QFrame):
    def __init__(self, parent=None):
        """Simulation controls."""
        # self.parent = parent
        super().__init__(parent)

        self.box = QtGui.QVBoxLayout()
        self.setLayout(self.box)

        # Select simulation
        self.simulations = OrderedDict(outdoor=outdoor,
                                       hallway=hallway,
                                       evacuation=evacuation)
        self.simu_names = ("",) + tuple(self.simulations.keys())

        simu_name = QComboBox()
        simu_name.addItems(self.simu_names)
        simu_name.currentIndexChanged[str].connect(self.setControls)

        # Timer for updating interactive plots
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.run)

        # Values
        self.simulation = None
        self.initialize = None
        self.values = OrderedDict()

    def updateValues(self):
        for key, value in self.values.items():
            self.values[key] = value.text()

    def initSimulation(self):
        if self.initialize is not None:
            self.updateValues()
            self.simulation = self.initialize(**self.values)

    def setControls(self, name):
        if name == "":
            # Clear current layout
            self.initialize = None
            self.simulation = None
        elif name in self.simu_names:
            agent_models = ("circular", "three_circle")
            body_types = ("adult", "male", "female", "child", "eldery")

            self.initialize = self.simulations[name]

            # Set arguments for selected simulation
            size = QSpinBox()
            size.setMinimum(1)
            size.setMaximum(1000)

            width = QDoubleSpinBox()
            width.setMinimum(0)

            height = QDoubleSpinBox()
            height.setMinimum(0)

            agent_model = QComboBox()
            agent_model.addItems(agent_models)

            body_type = QComboBox()
            body_type.addItems(body_types)

            self.value["size"] = size
            self.value["height"] = height
            self.value["width"] = width
            self.value["agent_model"] = agent_model
            self.value["body_type"] = body_type

            # Initialize simulation
            btn_initsimu = QtGui.QPushButton("Initialize simulation")
            btn_initsimu.clicked.connect(self.initSimulation)

            # Set visualizations
            density = QCheckBox("Density Grid")
            navigation = QCheckBox("Navigation Field")
            density.setEnabled(False)
            navigation.setEnabled(False)

            # Run simulation
            simulate = QtGui.QPushButton("Run Simulation")
            save = QtGui.QPushButton("Save")
            simulate.setEnabled(False)
            save.setEnabled(False)

            self.box.addItem(size)
            self.box.addItem(width)
            self.box.addItem(height)
            self.box.addItem(agent_model)
            self.box.addItem(body_type)
            self.box.addItem(btn_initsimu)
            self.box.addItem(density)
            self.box.addItem(navigation)
            self.box.addItem(simulate)
            self.box.addItem(save)


class MainWindow(QtGui.QWidget):
    # TODO: QMainWindow
    def __init__(self, parent=None):
        """Main window for gui application."""
        super(MainWindow, self).__init__(parent)

        pg.setConfigOptions(antialias=True)
        self.setWindowTitle("Crowd Dynamics")
        self.resize(1200, 800)

        # Timer for updating interactive plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.run)

        # Simulation Plot
        self.simulation = None
        self.simulation_name = None
        path = "/home/jaan/Dropbox/Projects/Crowd-Dynamics-Simulations/"
        # path2 = "/media/storage3/Crowd-Dynamics-Simulations/"
        self.simu_kw = {"size": 30, "width": 10, "height": 10, "path": path}
        self.graphics = pg.GraphicsLayoutWidget()
        self.central = SimulationGraphics()
        self.graphics.addItem(self.central, 0, 0, 1, 1)

        # Buttons for controlling the simulation
        self.btn_initsimu = QtGui.QPushButton("Initialize simulation")
        self.btn_initsimu.clicked.connect(self.initialize_simulation)

        self.btn_simulate = QtGui.QPushButton("Simulate")
        self.btn_simulate.setCheckable(True)
        self.btn_simulate.clicked.connect(self.simulate)

        btn4 = QtGui.QPushButton("Save and exit")
        btn4.clicked.connect(self.exit_and_save)

        # Layout for widgets
        layout = QtGui.QGridLayout()
        self.setLayout(layout)

        # row | col | rowspan | colspan
        layout.addWidget(self.graphics, 0, 1, 11, 5)

    def initialize_simulation(self):
        if self.simulation_name is not None:
            self.simulation = hallway(**self.simu_kw)
            self.central.setSimulation(self.simulation)

    def start(self):
        if not self.timer.isActive():
            print("Simulation started")
            self.timer.start(0)

    def stop(self):
        if self.timer.isActive():
            print("Simulation stopped")
            self.timer.stop()

    def simulate(self):
        if self.btn_simulate.isChecked():
            self.start()
        else:
            self.stop()

    def exit_and_save(self):
        if self.simulation is not None:
            self.stop()
            self.simulation.exit()
            self.close()

    def run(self):
        if self.simulation.advance():
            print("Simulation started")
            self.central.updateData()
        else:
            print("End of simulation")
            self.stop()
