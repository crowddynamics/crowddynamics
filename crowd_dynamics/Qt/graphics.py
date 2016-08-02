import numpy as np

import pyqtgraph as pg
from crowd_dynamics.simulation import Simulation
from crowd_dynamics.structure.area import Rectangle
from crowd_dynamics.structure.wall import LinearWall


class SimulationGraphics(pg.PlotItem):
    title = "Crowd Simulation"
    name = "simulation_graphics"

    # TODO: Remote processing
    # TODO: Legend
    # TODO: Coloring of agents (Forces, etc)

    def __init__(self):
        super(SimulationGraphics, self).__init__(name=self.name)

        # One to one scale for x and y coordinates
        self.setAspectLocked(lock=True, ratio=1)
        self.showGrid(True, True, 0.25)
        self.setLabels(title=self.title, left="y", bottom="x")

        # Data
        self.simulation = None

        # Brushes
        self.domain_brush = pg.mkBrush(255, 255, 255, 255 // 4)  # White, transparent
        self.impatient = pg.mkBrush(255, 0, 0, 255)  # RGBA
        self.patient = pg.mkBrush(0, 0, 255, 255)  # RGBA
        self.states = np.array((self.impatient, self.patient))

        # Order of initialization of plots matters here!
        # self.domain = self.plot()
        # self.goals = self.plot()

        # Agent
        self.left_shoulder = self.plot()
        self.right_shoulder = self.plot()
        self.torso = self.plot()
        self.direction = self.plot()

        # Walls
        self.walls = self.plot()

    def setSimulation(self, simulation: Simulation):
        self.simulation = simulation
        domain = self.simulation.domain
        agent = self.simulation.agent

        if domain is not None:
            if isinstance(domain, Rectangle):
                self.setRange(xRange=domain.x, yRange=domain.y)
                self.disableAutoRange()

        # Areas
        if domain is not None:
            if isinstance(domain, Rectangle):
                # c1 = pg.PlotDataItem([domain.x[0]], [domain.y[0]])
                # c2 = pg.PlotDataItem([domain.x[1]], [domain.y[1]])
                # pg.FillBetweenItem(c1, c2, brush=self.domain_brush)
                pass

        circle = lambda radius: dict(symbol='o',
                                     symbolSize=2 * radius,
                                     symbolBrush=self.patient,
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

        for wall in self.simulation.wall:
            if isinstance(wall, LinearWall):
                connect = np.zeros(2 * wall.size, dtype=np.int32)
                connect[::2] = np.ones(wall.size, dtype=np.int32)
                self.walls.setData(wall.params[:, :, 0].flatten(),
                                   wall.params[:, :, 1].flatten(),
                                   connect=connect)

        self.updateData()

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