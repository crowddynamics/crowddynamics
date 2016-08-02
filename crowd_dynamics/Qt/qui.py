import numpy as np
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore

from crowd_dynamics.simulation import Simulation
from crowd_dynamics.structure.area import Rectangle
from crowd_dynamics.structure.wall import LinearWall


class SimulationGraphics(pg.PlotItem):
    name = "central_item"
    title = "Simulation"

    def __init__(self, simulation: Simulation):
        """Central plot."""
        # TODO: Remote processing
        # TODO: Legend
        # TODO: Coloring of agents (Forces, etc)
        super(SimulationGraphics, self).__init__(name=self.name)

        # Data
        self.simulation = simulation
        agent = self.simulation.agent
        domain = self.simulation.domain

        # One to one scale for x and y coordinates
        self.setAspectLocked(lock=True, ratio=1)
        self.showGrid(True, True, 0.5)
        self.setLabels(title=self.title, left="y", bottom="x")

        if domain is not None:
            if isinstance(domain, Rectangle):
                self.setRange(xRange=domain.x, yRange=domain.y)
                self.disableAutoRange()

        # Areas
        if domain is not None:
            if isinstance(domain, Rectangle):
                # brush = pg.mkBrush(255, 255, 255, 255 // 4)  # White, transparent
                # c1 = pg.PlotDataItem([domain.x[0]], [domain.y[0]])
                # c2 = pg.PlotDataItem([domain.x[1]], [domain.y[1]])
                # pg.FillBetweenItem(c1, c2, brush=brush)
                pass
        # TODO: Goals

        # Agent
        self.impatient = pg.mkBrush(255, 0, 0, 255)  # RGBA
        self.patient = pg.mkBrush(0, 0, 255, 255)  # RGBA
        self.states = np.array((self.impatient, self.patient))

        self.left_shoulder = self.plot()
        self.right_shoulder = self.plot()
        self.torso = self.plot()
        self.direction = self.plot()

        self.setAgent()

        # Walls
        self.walls = self.plot()
        self.setWalls()

        self.updateData()

    def setWalls(self):
        for wall in self.simulation.wall:
            if isinstance(wall, LinearWall):
                connect = np.zeros(2 * wall.size, dtype=np.int32)
                connect[::2] = np.ones(wall.size, dtype=np.int32)
                self.walls.setData(wall.params[:, :, 0].flatten(),
                                   wall.params[:, :, 1].flatten(),
                                   connect=connect)

    def setAgent(self):
        agent = self.simulation.agent
        brush = pg.mkBrush(0, 0, 255, 255)
        circle = lambda radius: dict(symbol='o',
                                     symbolSize=2 * radius,
                                     symbolBrush=brush,
                                     pen=None,
                                     pxMode=False)
        if agent.circular:
            self.torso.setData(**circle(agent.radius))
        elif agent.three_circle:
            self.torso.setData(**circle(agent.r_t))
            self.left_shoulder.setData(**circle(agent.r_s))
            self.right_shoulder.setData(**circle(agent.r_s))

            connect = np.ones(3 * agent.size, dtype=np.int32)
            connect[2::3] = np.zeros(agent.size, dtype=np.int32)
            self.direction.setData(connect=connect)

    def updateData(self):
        """Updates data in the plot."""
        agent = self.simulation.agent

        if self.simulation.egress_model is not None:
            brush = self.states[self.simulation.egress_model.strategy]
        else:
            brush = self.patient

        self.torso.setData(agent.position, symbolBrush=brush)

        if agent.three_circle:
            self.left_shoulder.setData(agent.position_ls, symbolBrush=brush)
            self.right_shoulder.setData(agent.position_rs, symbolBrush=brush)

            array = np.concatenate((agent.position_ls, agent.front, agent.position_rs), axis=1)
            array = array.reshape(3 * agent.shape[0], agent.shape[1])
            self.direction.setData(array)

        text = "Iterations: {} " \
               "Simulation time: {:0.2f} " \
               "Agents in goal: {}"
        stats = self.simulation.result
        self.setLabels(top=text.format(stats.iterations, stats.simulation_time,
                                       stats.in_goal))


class Main(pg.LayoutWidget):
    def __init__(self, simulation: Simulation=None, parent=None):
        super(Main, self).__init__(parent)

        pg.setConfigOptions(antialias=True)
        self.setWindowTitle("Crowd Dynamics")
        self.resize(1200, 800)

        # Timer for updating interactive plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.run)

        # Simulation Plot
        self.simulation = simulation
        self.graphics = pg.GraphicsLayoutWidget()
        if self.simulation is None:
            self.central = pg.PlotItem()
        else:
            self.central = SimulationGraphics(self.simulation)
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
        lbl1 = QtGui.QLabel("Visualization")
        cbx1 = QtGui.QCheckBox("Density")
        cbx2 = QtGui.QCheckBox("Potential field")

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

        self.addWidget(lbl1, row=4, col=0)
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
        simulation = initialize(size=30, width=10, height=10)
        central = SimulationGraphics(simulation)
        self.central.close()
        self.graphics.addItem(central, 0, 0, 1, 1)
        self.simulation = simulation
        self.central = central

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


def main(simulation: Simulation=None):
    """Launches Qt application for visualizing simulation.
    :param simulation:
    """
    # TODO: Read simulation data from hdf5 file
    # TODO: MoviePy
    import sys

    app = QtGui.QApplication(sys.argv)
    qui = Main(simulation)

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(app.exec_())
    else:
        raise Warning("Interactive mode or Pyside are not supported.")


# if __name__ == '__main__':
#     main()
