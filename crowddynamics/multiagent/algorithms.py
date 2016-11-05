import numpy as np

from crowddynamics.core.interactions import agent_agent_block_list, agent_wall
from crowddynamics.core.motion import integrate, force_fluctuation, \
    torque_fluctuation, force_adjust, torque_adjust

from crowddynamics.core.navigation import distance_map, direction_map, \
    _to_indices
from crowddynamics.core.vector2D import angle_nx2
from crowddynamics.functions import public, timed
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

    @timed("Agent-Agent Interaction")
    def update(self):
        # agent_agent(self.simulation.agent)
        agent_agent_block_list(self.simulation.agent)


@public
class AgentObstacleInteractions(TaskNode):
    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    @timed("Agent-Obstacle Interaction")
    def update(self):
        agent_wall(self.simulation.agent, self.simulation.walls)


@public
class Navigation(TaskNode):
    """Determining target direction of an agent in multi-agent simulation.

    Algorithm based on solving the continous shortest path
    problem by solving eikonal equation. [1]_, [2]_

    There are at least two open source eikonal solvers. Fast marching method
    (FMM) [3]_ for rectangular and tetrahedral meshes using Python and C++ and
    fast iterative method (FIM) [4]_ for triangular meshes using c++ and CUDA.

    In this implementation we use the FMM algorithm because it is simpler.

    .. [1] Kretz, T., Große, A., Hengst, S., Kautzsch, L., Pohlmann, A., & Vortisch, P. (2011). Quickest Paths in Simulations of Pedestrians. Advances in Complex Systems, 14(5), 733–759. http://doi.org/10.1142/S0219525911003281
    .. [2] Cristiani, E., & Peri, D. (2015). Handling obstacles in pedestrian simulations: Models and optimization. Retrieved from http://arxiv.org/abs/1512.08528
    .. [3] https://github.com/scikit-fmm/scikit-fmm
    .. [4] https://github.com/SCIInstitute/SCI-Solver_Eikonal
    """

    # TODO: take into account radius of the agents

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
        self.distance_map = None
        self.direction_map = None

        if algorithm == "static":
            # self.static_potential()
            mgrid, dmap, phi = distance_map(
                self.simulation.domain,
                self.simulation.obstacles,
                self.simulation.exits,
                self.step
            )
            self.distance_map = dmap
            self.direction_map = direction_map(dmap)
        elif algorithm == "dynamic":
            raise NotImplementedError
        else:
            pass

    @timed("Navigation Time")
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

    @timed("Orientation Time")
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
