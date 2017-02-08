import numpy as np

from crowddynamics.core.geometry import shapes_to_point_pairs
from crowddynamics.core.integrator import euler_integration
from crowddynamics.core.interactions.interactions import agent_agent_block_list, agent_wall
from crowddynamics.core.motion import force_fluctuation, \
    force_adjust, torque_adjust, torque_fluctuation
from crowddynamics.core.steering.navigation import to_indices, static_potential
from crowddynamics.core.vector2D.vector2D import angle_nx2
from crowddynamics.io import HDFStore
from crowddynamics.taskgraph import TaskNode


class Integrator(TaskNode):
    r"""Integrator

    Attributes:
        simulation (MultiAgentSimulation):
            Simulation class

        dt (tuple[float]):
            Tuple of minimum and maximum timestep (dt_min, dt_max).
    """

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation
        self.dt = (0.001, 0.01)
        self.time_tot = np.float64(0.0)
        self.dt_prev = np.float64(np.nan)

    def update(self):
        self.dt_prev = euler_integration(self.simulation.agent, self.dt[0], self.dt[0])
        self.time_tot += self.dt_prev


class Fluctuation(TaskNode):
    r"""Fluctuation"""

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    def update(self):
        agent = self.simulation.agent
        i = agent.indices()

        agent.force[i] = force_fluctuation(agent.mass[i], agent.std_rand_force[i])

        if agent.orientable:
            agent.torque[i] = torque_fluctuation(agent.inertia_rot[i],
                                                 agent.std_rand_torque[i])


class Adjusting(TaskNode):
    r"""Adjusting"""

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    def update(self):
        agent = self.simulation.agent
        i = agent.indices()

        agent.force[i] = force_adjust(agent.mass[i],
                                      agent.tau_adj[i],
                                      agent.target_velocity[i],
                                      agent.target_direction[i],
                                      agent.velocity[i])
        if agent.orientable:
            agent.torque[i] = torque_adjust(agent.inertia_rot[i],
                                            agent.tau_rot[i],
                                            agent.target_orientation[i],
                                            agent.orientation[i],
                                            agent.target_angular_velocity[i],
                                            agent.angular_velocity[i])


class AgentAgentInteractions(TaskNode):
    r"""AgentAgentInteractions"""

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    def update(self):
        agent_agent_block_list(self.simulation.agent)


class AgentObstacleInteractions(TaskNode):
    r"""AgentObstacleInteractions"""

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

        # TODO: Expects that field is set prior to initialisation
        self.walls = shapes_to_point_pairs(self.simulation.obstacles)

    def update(self):
        agent_wall(self.simulation.agent, self.walls)


class Navigation(TaskNode):
    r"""Handles navigation in multi-agent simulation.

    Attributes:
        simulation:
        algorithm:
        step (float): Step size for the grid.

    """

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

        self.step = 0.01
        self.direction_map = None
        self.algorithm = "static"

        if self.algorithm == "static":
            self.direction_map = static_potential(self.step,
                                                  self.simulation.domain,
                                                  self.simulation.targets,
                                                  self.simulation.obstacles,
                                                  radius=0.3,
                                                  value=0.3)
        elif self.algorithm == "dynamic":
            raise NotImplementedError
        else:
            pass

    def update(self):
        i = self.simulation.agent.indices()
        points = self.simulation.agent.position[i]
        # indices = self.points_to_indices(points)
        indices = to_indices(points, self.step)
        indices = np.fliplr(indices)
        # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        # TODO: Handle index out of bounds -> numpy.take
        d = self.direction_map[indices[:, 0], indices[:, 1], :]
        self.simulation.agent.target_direction[i] = d


class Orientation(TaskNode):
    r"""Target orientation"""

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    def update(self):
        if self.simulation.agent.orientable:
            dir_to_orient = angle_nx2(self.simulation.agent.target_direction)
            self.simulation.agent.target_orientation[:] = dir_to_orient


class ExitSelection(TaskNode):
    """Exit selection policy."""

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    def update(self):
        pass


class Reset(TaskNode):
    r"""Reset"""

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    def update(self):
        self.simulation.agent.reset_motion()
        # self.agent.reset_neighbor()


class IO(TaskNode):
    r"""IO"""

    def __init__(self, simulation, structs, attribute_list):
        super().__init__()
        self.simulation = simulation
        self.hdfstore = HDFStore(self.simulation.name)
        # TODO: add struct(s) and attributes(s)
        for struct, attributes in zip(structs, attribute_list):
            self.hdfstore.add_dataset(struct, attributes)
            self.hdfstore.add_buffers(struct, attributes)
        self.iterations = 0

    def update(self, frequency=100):
        self.hdfstore.update_buffers()
        if self.iterations % frequency == 0:
            self.hdfstore.dump_buffers()
        self.iterations += 1


class GuiCommunication(TaskNode):
    """GuiCommunication"""
    pass


class Contains(TaskNode):
    """Contains"""
    pass
