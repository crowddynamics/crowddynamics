import numpy as np

from crowddynamics.core.interactions import agent_agent_block_list, agent_wall
from crowddynamics.core.motion import integrate, force_fluctuation, \
    torque_fluctuation, force_adjust, torque_adjust
from crowddynamics.core.navigation import _to_indices, static_potential
from crowddynamics.core.vector2D import angle_nx2
from crowddynamics.functions import public, Timed
from crowddynamics.task_graph import TaskNode


@public
class Integrator(TaskNode):
    def __init__(self, simulation, dt):
        """

        :param simulation: Simulation class
        :param dt: Tuple of minumum and maximum timestep (dt_min, dt_max).
        """
        super().__init__()

        self.simulation = simulation
        self.dt = dt

        self.time_tot = np.float64(0)
        self.dt_prev = np.float64(np.nan)

    def update(self):
        """
        Integrates the system.

        Returns:
            None

        """
        self.dt_prev = integrate(self.simulation.agent, *self.dt)
        self.time_tot += self.dt_prev
        self.simulation.dt_prev = self.dt_prev
        self.simulation.time_tot += self.dt_prev


@public
class Fluctuation(TaskNode):
    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    def update(self):
        force_fluctuation(self.simulation.agent)
        torque_fluctuation(self.simulation.agent)


@public
class Adjusting(TaskNode):
    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    def update(self):
        force_adjust(self.simulation.agent)
        torque_adjust(self.simulation.agent)


@public
class AgentAgentInteractions(TaskNode):
    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    @Timed("Agent-Agent Interaction")
    def update(self):
        # agent_agent(self.simulation.agent)
        agent_agent_block_list(self.simulation.agent)


@public
class AgentObstacleInteractions(TaskNode):
    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    @Timed("Agent-Obstacle Interaction")
    def update(self):
        agent_wall(self.simulation.agent, self.simulation.walls)


@public
class Navigation(TaskNode):
    """
    Handles navigation in multi-agent simulation.
    """

    def __init__(self, simulation, algorithm="static", step=0.01):
        """

        Args:
            simulation:
            algorithm:
            step (float): Step size for the grid.

        """
        super().__init__()
        self.simulation = simulation

        self.step = step
        self.direction_map = None

        if algorithm == "static":
            self.direction_map = static_potential(self.step,
                                                  self.simulation.domain,
                                                  self.simulation.exits,
                                                  self.simulation.obstacles,
                                                  radius=0.3,
                                                  value=0.3)
        elif algorithm == "dynamic":
            raise NotImplementedError
        else:
            pass

    @Timed("Navigation Time")
    def update(self):
        """
        Changes target directions of active agents.

        Returns:
            None.

        """
        i = self.simulation.agent.indices()
        points = self.simulation.agent.position[i]
        # indices = self.points_to_indices(points)
        indices = _to_indices(points, self.step)
        indices = np.fliplr(indices)
        # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        # TODO: Handle index out of bounds -> numpy.take
        d = self.direction_map[indices[:, 0], indices[:, 1], :]
        self.simulation.agent.target_direction[i] = d


@public
class Orientation(TaskNode):
    """
    Target orientation
    """

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    @Timed("Orientation Time")
    def update(self):
        if self.simulation.agent.orientable:
            dir_to_orient = angle_nx2(self.simulation.agent.target_direction)
            self.simulation.agent.target_angle[:] = dir_to_orient


@public
class ExitSelection(TaskNode):
    """Exit selection policy."""

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    def update(self):
        pass
