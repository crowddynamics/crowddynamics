from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QComboBox, QIntValidator, QLineEdit, QCheckBox, \
    QDoubleValidator

import pyqtgraph as pg
from crowd_dynamics.Qt.graphics import SimulationGraphics


class Gui(pg.LayoutWidget):
    def __init__(self, parent=None):
        super(Gui, self).__init__(parent)

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

        # Check boxes for optional data visualization
        # Density, flow rate, pressure, potential field, forces
        cbx1 = QCheckBox("Density Grid")
        cbx2 = QCheckBox("Navigation Field")

        # Button popup menus
        # simulation type, body type, agent model, attributes to save
        simu_names = ("Select Simulation", "outdoor", "hallway", "evacuation")
        agent_models = ("circular", "three_circle")
        body_types = ("adult", "male", "female", "child", "eldery")

        menu1 = QComboBox()
        menu1.addItems(simu_names)
        menu1.currentIndexChanged[str].connect(self.select_simulation)

        menu2 = QComboBox()
        menu2.addItems(agent_models)
        menu2.currentIndexChanged[str].connect(self.select_agent_model)

        menu3 = QComboBox()
        menu3.addItems(body_types)
        menu3.currentIndexChanged[str].connect(self.select_body_type)

        # Text boxes for setting values
        # Size
        self.line1 = QLineEdit()
        self.line1.setMaxLength(4)
        self.line1.setMaximumWidth(150)
        self.line1.setValidator(QIntValidator())
        self.line1.returnPressed.connect(self.set_size)

        # Width
        self.line2 = QLineEdit()
        self.line2.setMaximumWidth(150)
        self.line2.setValidator(QDoubleValidator())
        self.line2.returnPressed.connect(self.set_width)

        # Height
        self.line3 = QLineEdit()
        self.line3.setMaximumWidth(150)
        self.line3.setValidator(QDoubleValidator())
        self.line3.returnPressed.connect(self.set_height)

        # Layout for widgets
        self.addWidget(self.graphics, row=0, col=1, rowspan=11, colspan=5)

        self.addWidget(self.btn_simulate, row=12, col=1)
        self.addWidget(btn4, row=12, col=2)

        self.addWidget(menu1, row=0, col=0)
        self.addWidget(menu2, row=1, col=0)
        self.addWidget(menu3, row=2, col=0)

        self.addWidget(self.line1, row=3, col=0)
        self.addWidget(self.line2, row=4, col=0)
        self.addWidget(self.line3, row=5, col=0)

        self.addWidget(self.btn_initsimu, row=6, col=0)

        self.addWidget(cbx1, row=7, col=0)
        self.addWidget(cbx2, row=8, col=0)

        # Display the window when initialized
        self.show()

    def initialize_simulation(self):
        name = self.simulation_name
        if name == "outdoor":
            from crowd_dynamics.examples.outdoor import initialize
        elif name == "hallway":
            from crowd_dynamics.examples.hallway import initialize
        elif name == "evacuation":
            from crowd_dynamics.examples.evacuation import initialize
        else:
            raise ValueError("")
        self.simulation = initialize(**self.simu_kw)
        self.central.setSimulation(self.simulation)

    def select_simulation(self, name):
        print("Simulation:", name)
        self.simulation_name = name

    def select_agent_model(self, model):
        print("Agent model:", model)
        self.simu_kw["model"] = model

    def select_body_type(self, btype):
        print("Body type:", btype)
        self.simu_kw["body_type"] = btype

    def set_size(self):
        size = int(self.line1.text())
        if size >= 1:
            self.simu_kw["size"] = size
            self.line1.clear()

    def set_width(self):
        width = float(self.line2.text())
        if width > 0:
            self.simu_kw["width"] = width
            self.line2.clear()

    def set_height(self):
        height = float(self.line3.text())
        if height > 0:
            self.simu_kw["height"] = height
            self.line3.clear()

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
