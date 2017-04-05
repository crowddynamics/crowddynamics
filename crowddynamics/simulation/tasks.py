import numpy as np
from matplotlib.path import Path

from crowddynamics.core.geometry import shapes_to_point_pairs
from crowddynamics.core.integrator import euler_integration
from crowddynamics.core.interactions.interactions import \
    agent_agent_block_list, agent_wall
from crowddynamics.core.motion.adjusting import force_adjust, torque_adjust
from crowddynamics.core.motion.fluctuation import force_fluctuation, \
    torque_fluctuation
from crowddynamics.core.steering.navigation import to_indices, static_potential
from crowddynamics.core.structures.agents import agent_type_three_circle
from crowddynamics.core.vector.vector2D import angle_nx2
from crowddynamics.io import HDFStore, Record
from crowddynamics.simulation.taskgraph import TaskNode

# TODO: Integrator: time-limit, iterations limit -> signal
# TODO: Contains:   diff limit


__all__ = """
Integrator
Fluctuation
Adjusting
AgentAgentInteractions
AgentObstacleInteractions
Navigation
Orientation
ExitSelection
Reset
HDFNode
Contains
""".split()


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

    def set(self, iter_limit=None, time_limit=None):
        pass

    def signal(self):
        pass

    def update(self):
        self.dt_prev = euler_integration(self.simulation.agent, self.dt[0], self.dt[1])
        self.time_tot += self.dt_prev


class Fluctuation(TaskNode):
    r"""Fluctuation"""

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    def update(self):
        agent = self.simulation.agent

        agent['force'] += force_fluctuation(agent['mass'],
                                            agent['std_rand_force'])

        if agent.orientable:
            agent['torque'] += torque_fluctuation(agent['inertia_rot'],
                                                  agent['std_rand_torque'])


class Adjusting(TaskNode):
    r"""Adjusting"""

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    def update(self):
        agent = self.simulation.agent

        agent['force'] += force_adjust(agent['mass'],
                                       agent['tau_adj'],
                                       agent['target_velocity'],
                                       agent['target_direction'],
                                       agent['velocity'])
        if agent.orientable:
            agent['torque'] += torque_adjust(agent['inertia_rot'],
                                             agent['tau_rot'],
                                             agent['target_orientation'],
                                             agent['orientation'],
                                             agent['target_angular_velocity'],
                                             agent['angular_velocity'])


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
        self.simulation.agent['target_direction'] = d


class Orientation(TaskNode):
    r"""Target orientation"""

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    def update(self):
        if self.simulation.agent.dtype is agent_type_three_circle:
            dir_to_orient = angle_nx2(self.simulation.agent.target_direction)
            self.simulation.agent['target_orientation'] = dir_to_orient


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


class HDFNode(TaskNode):
    r"""Saves data to hdf5 file.

    - Agent data
    - Game data
    - etc
    """

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation
        self.hdfstore = HDFStore(self.simulation.name)
        self.iterations = 0

    def set(self, records):
        if isinstance(records, Record):
            self.hdfstore.add_dataset(records)
        else:
            for record in records:
                self.hdfstore.add_dataset(record)

    def update(self, frequency=100):
        self.iterations += 1
        self.hdfstore.update_buffers()
        if self.iterations % frequency == 0:
            self.hdfstore.dump_buffers()


class Contains(TaskNode):
    """Contains"""

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

        self.path = None
        self.inside = np.zeros(self.simulation.agent.size, np.bool8)

    def set(self, polygon):
        self.path = Path(np.asarray(polygon.exterior))
        self.update()

    def update(self, *args, **kwargs):
        position = self.simulation.agent.position
        inside = self.path.contains_points(position)
        # out: True  -> False
        # in:  False -> True
        changed = self.inside ^ inside
        self.inside = inside
        diff = np.sum(changed)
