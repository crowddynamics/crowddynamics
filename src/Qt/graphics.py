import numpy as np

import pyqtgraph as pg
from src.simulation.multiagent import MultiAgentSimulation
from src.structure.area import Rectangle
from src.structure.obstacle import LinearWall


class SimulationPlot(pg.PlotItem):
    def __init__(self):
        """Widget for displaying simulation graphics."""
        super(SimulationPlot, self).__init__()

        # Data
        self.simulation = None

        # Plot settings
        self.setAspectLocked(lock=True, ratio=1)  # One to one scale
        self.showGrid(True, True, 0.25)
        self.disableAutoRange()

        # Pens
        self.active_pen = pg.mkPen('w')
        self.inactive_pen = pg.mkPen(None)

        # Brushes: RGBA
        self.domain_brush = pg.mkBrush(255, 255, 255, 255 // 8)  # White, transparent
        self.goal_brush = pg.mkBrush(255, 255, 255, 255 // 4)    # White, transparent
        self.inactive = pg.mkBrush(0, 0, 0, 0)                   # Fully transparent
        self.impatient = pg.mkBrush(255, 0, 0, 255)
        self.patient = pg.mkBrush(0, 0, 255, 255)
        self.states = np.array((self.impatient, self.patient))

        # Order of initialization of plots matters here!
        self.left_shoulder = self.plot()
        self.right_shoulder = self.plot()
        self.torso = self.plot()
        self.direction = self.plot()
        self.walls = self.plot()

    def set_simulation(self, simulation: MultiAgentSimulation):
        # Clear plots
        self.clearPlots()
        self.clear()

        # Initialize plots
        self.left_shoulder = self.plot()
        self.right_shoulder = self.plot()
        self.torso = self.plot()
        self.direction = self.plot()
        self.walls = self.plot()

        # Initialise data
        self.simulation = simulation
        self.init_data()

    def init_data(self):
        domain = self.simulation.domain
        goals = self.simulation.goals
        agent = self.simulation.agent

        if domain is not None:
            if isinstance(domain, Rectangle):
                self.setRange(xRange=domain.x, yRange=domain.y)
                c1 = pg.PlotDataItem(domain.x, (domain.y[0], domain.y[0]))
                c2 = pg.PlotDataItem(domain.x, (domain.y[1], domain.y[1]))
                fill = pg.FillBetweenItem(c1, c2, brush=self.domain_brush)
                self.addItem(fill)

        if goals is not None:
            for goal in goals:
                if isinstance(goal, Rectangle):
                    c1 = pg.PlotDataItem(goal.x, (goal.y[0], goal.y[0]))
                    c2 = pg.PlotDataItem(goal.x, (goal.y[1], goal.y[1]))
                    fill = pg.FillBetweenItem(c1, c2, brush=self.goal_brush)
                    self.addItem(fill)

        circle = lambda radius: dict(symbol='o',
                                     symbolSize=2 * radius,
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

        for wall in self.simulation.walls:
            if isinstance(wall, LinearWall):
                connect = np.zeros(2 * wall.size, dtype=np.int32)
                connect[::2] = np.ones(wall.size, dtype=np.int32)
                self.walls.setData(wall.params[:, :, 0].flatten(),
                                   wall.params[:, :, 1].flatten(),
                                   connect=connect)

        self.update_data()

    def update_data(self):
        """Updates data in the plot."""
        if self.simulation is not None:
            agent = self.simulation.agent
            sympen = np.zeros(agent.size, dtype=object)
            sympen[:] = self.active_pen
            sympen[agent.active ^ True] = self.inactive_pen

            if self.simulation.egress_model is not None:
                brush = self.states[self.simulation.egress_model.strategy]
            else:
                brush = np.zeros(agent.size, dtype=object)
                brush[:] = self.patient

            brush[agent.active ^ True] = self.inactive

            self.torso.setData(agent.position,
                               symbolBrush=brush,
                               symbolPen=sympen)

            if agent.three_circle:
                self.left_shoulder.setData(agent.position_ls,
                                           symbolBrush=brush,
                                           symbolPen=sympen)
                self.right_shoulder.setData(agent.position_rs,
                                            symbolBrush=brush,
                                            symbolPen=sympen)

                array = np.concatenate((agent.position_ls, agent.front, agent.position_rs), axis=1)
                array = array.reshape(3 * agent.shape[0], agent.shape[1])
                self.direction.setData(array)  # TODO: pen
