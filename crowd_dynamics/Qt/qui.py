from PyQt4 import QtGui, QtCore

import pyqtgraph as pg
from crowd_dynamics.Qt.graphics import SimulationGraphics
from crowd_dynamics.simulation import Simulation


class Gui(pg.LayoutWidget):
    def __init__(self, simulation: Simulation=None, parent=None):
        super(Gui, self).__init__(parent)

        pg.setConfigOptions(antialias=True)
        self.setWindowTitle("Crowd Dynamics")
        self.resize(1200, 800)

        # Timer for updating interactive plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.run)

        # Simulation Plot
        self.simulation = simulation
        self.graphics = pg.GraphicsLayoutWidget()
        self.central = SimulationGraphics()
        self.graphics.addItem(self.central, 0, 0, 1, 1)

        # Buttons for controlling the simulation
        btn1 = QtGui.QPushButton("Initialize simulation")
        # btn1.clicked.connect()

        btn2 = QtGui.QPushButton("Start")
        btn2.clicked.connect(self.start)

        btn3 = QtGui.QPushButton("Stop")
        btn3.clicked.connect(self.stop)

        btn4 = QtGui.QPushButton("Save and exit")
        btn4.clicked.connect(self.exit_and_save)

        # Check boxes for optional data visualization
        # Density, flow rate, pressure, potential field, forces
        # lbl1 = QtGui.QLabel("Visualization")
        cbx1 = QtGui.QCheckBox("Density Grid")
        cbx2 = QtGui.QCheckBox("Navigation Field")

        # Button popup menus
        # simulation type, body type, agent model, attributes to save
        simu_names = ("outdoor", "hallway", "evacuation")
        agent_models = ("circular", "three_circle")
        body_types = ("adult", "male", "female", "child", "eldery")

        menu1 = QtGui.QComboBox()
        menu1.addItems(simu_names)
        menu1.currentIndexChanged[str].connect(self.select_simulation)

        menu2 = QtGui.QComboBox()
        menu2.addItems(agent_models)
        menu2.currentIndexChanged[str].connect(self.select_agent_model)

        menu3 = QtGui.QComboBox()
        menu3.addItems(body_types)
        menu3.currentIndexChanged[str].connect(self.select_body_type)

        # Text boxes for setting values
        line1 = QtGui.QLineEdit()
        line1.setMaximumWidth(150)

        # Layout for widgets
        self.addWidget(self.graphics, row=0, col=1, rowspan=11, colspan=5)

        self.addWidget(btn1, row=12, col=1)
        self.addWidget(btn2, row=12, col=2)
        self.addWidget(btn3, row=12, col=3)
        self.addWidget(btn4, row=12, col=4)

        self.addWidget(menu1, row=0, col=0)
        self.addWidget(menu2, row=1, col=0)
        self.addWidget(menu3, row=2, col=0)
        self.addWidget(line1, row=3, col=0)

        # self.addWidget(lbl1, row=4, col=0)
        self.addWidget(cbx1, row=5, col=0)
        self.addWidget(cbx2, row=6, col=0)

        # Display the window when initialized
        self.show()

    def select_simulation(self, name):
        print(name)
        if name == "outdoor":
            from crowd_dynamics.examples.outdoor import initialize
        elif name == "hallway":
            from crowd_dynamics.examples.hallway import initialize
        elif name == "evacuation":
            from crowd_dynamics.examples.evacuation import initialize
        else:
            raise ValueError("")

        path = "/home/jaan/Dropbox/Projects/Crowd-Dynamics-Simulations/"
        self.simulation = initialize(size=30, width=10, height=10, path=path)
        self.central.setSimulation(self.simulation)

    def select_agent_model(self, name):
        pass

    def select_body_type(self, name):
        pass

    def start(self):
        if not self.timer.isActive():
            self.timer.start(0)

    def stop(self):
        if self.timer.isActive():
            print("Simulation stopped")
            self.timer.stop()

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
