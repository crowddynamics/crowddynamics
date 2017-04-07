import numpy as np
from matplotlib.path import Path

from crowddynamics.core.integrator import euler_integration
from crowddynamics.core.interactions.interactions import \
    circular_agent_linear_wall, three_circle_agent_linear_wall, \
    agent_agent_block_list_circular, agent_agent_block_list_three_circle
from crowddynamics.core.motion.adjusting import force_adjust, torque_adjust
from crowddynamics.core.motion.fluctuation import force_fluctuation, \
    torque_fluctuation
from crowddynamics.core.steering.navigation import to_indices, static_potential
from crowddynamics.core.structures.agents import reset_motion, is_model
from crowddynamics.core.vector.vector2D import angle_nx2
from crowddynamics.io import HDFStore, Record
from crowddynamics.simulation.multiagent import MultiAgentSimulation
from crowddynamics.simulation.taskgraph import TaskNode

__all__ = """
MASTaskNode
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


class MASTaskNode(TaskNode):
    def __init__(self, simulation):
        super(MASTaskNode, self).__init__()
        assert isinstance(simulation, MultiAgentSimulation)
        self.simulation = simulation


class Integrator(MASTaskNode):
    r"""Integrator"""

    def __init__(self, simulation):
        super().__init__(simulation)
        self.dt = (0.001, 0.01)
        self.time_tot = np.float64(0.0)
        self.dt_prev = np.float64(np.nan)

    def set(self, iter_limit=None, time_limit=None):
        pass

    def signal(self):
        pass

    def update(self):
        self.dt_prev = euler_integration(self.simulation.agents_array,
                                         self.dt[0], self.dt[1])
        self.time_tot += self.dt_prev


class Fluctuation(MASTaskNode):
    r"""Fluctuation"""

    def update(self):
        agent = self.simulation.agents_array
        agent['force'] += force_fluctuation(agent['mass'],
                                            agent['std_rand_force'])
        if agent.orientable:
            agent['torque'] += torque_fluctuation(agent['inertia_rot'],
                                                  agent['std_rand_torque'])


class Adjusting(MASTaskNode):
    r"""Adjusting"""

    def update(self):
        agent = self.simulation.agents_array
        agent['force'] += force_adjust(
            agent['mass'], agent['tau_adj'], agent['target_velocity'],
            agent['target_direction'], agent['velocity'])
        if agent.orientable:
            agent['torque'] += torque_adjust(
                agent['inertia_rot'], agent['tau_rot'],
                agent['target_orientation'], agent['orientation'],
                agent['target_angular_velocity'], agent['angular_velocity'])


class AgentAgentInteractions(MASTaskNode):
    r"""AgentAgentInteractions"""

    def update(self):
        if is_model(self.simulation.agents_array, 'circular'):
            agent_agent_block_list_circular(self.simulation.agents_array)
        elif is_model(self.simulation.agents_array, 'three_circle'):
            agent_agent_block_list_three_circle(self.simulation.agents_array)


class AgentObstacleInteractions(MASTaskNode):
    r"""AgentObstacleInteractions"""

    def update(self):
        if is_model(self.simulation.agents_array, 'circular'):
            circular_agent_linear_wall(self.simulation.agents_array,
                                       self.simulation.obstacles_array)
        elif is_model(self.simulation.agents_array, 'three_circle'):
            three_circle_agent_linear_wall(self.simulation.agents_array,
                                           self.simulation.obstacles_array)


class Navigation(MASTaskNode):
    r"""Handles navigation in multi-agent simulation."""

    def __init__(self, simulation):
        super().__init__(simulation)
        self.step = 0.01
        self.direction_map = static_potential(self.step,
                                              self.simulation.domain,
                                              self.simulation.targets,
                                              self.simulation.obstacles,
                                              radius=0.3,
                                              value=0.3)

    def update(self):
        points = self.simulation.agents_array['position']
        indices = to_indices(points, self.step)
        indices = np.fliplr(indices)
        # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        # TODO: Handle index out of bounds -> numpy.take
        d = self.direction_map[indices[:, 0], indices[:, 1], :]
        self.simulation.agents_array['target_direction'] = d


class Orientation(MASTaskNode):
    r"""Target orientation"""

    def update(self):
        if is_model(self.simulation.agents_array, 'three_circle'):
            dir_to_orient = angle_nx2(self.simulation.agents_array['target_direction'])
            self.simulation.agents_array['target_orientation'] = dir_to_orient


class ExitSelection(MASTaskNode):
    """Exit selection policy."""

    def update(self):
        pass


class Reset(MASTaskNode):
    r"""Reset"""

    def update(self):
        reset_motion(self.simulation.agents_array)
        # TODO: reset agent neighbor


class HDFNode(MASTaskNode):
    r"""Saves data to hdf5 file."""

    def __init__(self, simulation):
        super().__init__(simulation)
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


class Contains(MASTaskNode):
    """Contains"""

    def __init__(self, simulation):
        super().__init__(simulation)

        self.path = None
        self.inside = np.zeros(self.simulation.agents.size, np.bool8)

    def set(self, polygon):
        self.path = Path(np.asarray(polygon.exterior))
        self.update()

    def update(self, *args, **kwargs):
        position = self.simulation.agents_array['position']
        inside = self.path.contains_points(position)
        # out: True  -> False
        # in:  False -> True
        changed = self.inside ^ inside
        self.inside = inside
        diff = np.sum(changed)
