import os

import numpy as np
from matplotlib.path import Path
from traitlets.traitlets import Float

from crowddynamics.core.geometry import geom_to_linear_obstacles
from crowddynamics.core.integrator import velocity_verlet_integrator
from crowddynamics.core.interactions.interactions import agent_agent_block_list, \
    agent_obstacle
from crowddynamics.core.motion.adjusting import force_adjust_agents, \
    torque_adjust_agents
from crowddynamics.core.motion.fluctuation import force_fluctuation, \
    torque_fluctuation
from crowddynamics.core.steering.navigation import navigation, herding
from crowddynamics.core.steering.orientation import \
    orient_towards_target_direction
from crowddynamics.io import save_npy, save_csv
from crowddynamics.simulation.agents import reset_motion, is_model
from crowddynamics.simulation.base import LogicNodeBase


class LogicNode(LogicNodeBase):
    """Simulation logic is programmed as a tree of dependencies of the order of
    the execution. For example simulation's logic tree could look like::

        Reset
        └── Integrator
            ├── Fluctuation
            ├── Adjusting
            │   ├── Navigation
            │   └── Orientation
            ├── AgentAgentInteractions
            └── AgentObstacleInteractions

    In this tree we can notice the dependencies. For example before using
    updating `Adjusting` node we need to update `Navigation` and `Orientation`
    nodes.
    """

    def __init__(self, simulation):
        super(LogicNode, self).__init__()
        self.simulation = simulation

    def update(self, *args, **kwargs):
        raise NotImplementedError


# Motion

class Reset(LogicNode):
    def update(self):
        reset_motion(self.simulation.agents.array)
        # TODO: reset agent neighbor


class Integrator(LogicNode):
    dt_min = Float(0.001, min=0)
    dt_max = Float(0.010, min=0)

    def update(self):
        dt = velocity_verlet_integrator(self.simulation.agents.array,
                                        self.dt_min, self.dt_max)
        self.simulation.data['dt'] = dt
        self.simulation.data['time_tot'] += dt


class Fluctuation(LogicNode):
    def update(self):
        agents = self.simulation.agents.array
        agents['force'] += force_fluctuation(agents['mass'],
                                             agents['std_rand_force'])
        if is_model(agents, 'three_circle'):
            agents['torque'] += torque_fluctuation(agents['inertia_rot'],
                                                   agents['std_rand_torque'])


class Adjusting(LogicNode):
    def update(self):
        agents = self.simulation.agents.array
        force_adjust_agents(agents)
        if is_model(agents, 'three_circle'):
            torque_adjust_agents(agents)


class AgentAgentInteractions(LogicNode):
    sight_soc = Float(
        default_value=3.0, min=0,
        help='')
    f_soc_max = Float(
        default_value=2e3, min=0,
        help='')

    def update(self):
        agent_agent_block_list(self.simulation.agents.array)


class AgentObstacleInteractions(LogicNode):
    def update(self):
        agents = self.simulation.agents.array
        obstacles = geom_to_linear_obstacles(self.simulation.field.obstacles)
        agent_obstacle(agents, obstacles)


# Steering

class Navigation(LogicNode):
    step = Float(0.02, min=0)
    radius = Float(0.5, min=0)
    strength = Float(0.3, min=0, max=1)

    def update(self):
        agents = self.simulation.agents.array
        targets = agents['target']
        for target in set(targets):
            mgrid, distance_map, direction_map = \
                self.simulation.field.navigation_to_target(
                    target, self.step, self.radius, self.strength)
            navigation(agents, targets == target, mgrid, direction_map)


class Herding(LogicNode):
    sight_herding = Float(default_value=3.0, min=0)

    def update(self, *args, **kwargs):
        agents = self.simulation.agents.array
        herding(agents, agents['herding'], self.sight_herding)


class Orientation(LogicNode):
    def update(self):
        if is_model(self.simulation.agents.array, 'three_circle'):
            orient_towards_target_direction(self.simulation.agents.array)


# IO

def _save_condition(simulation, frequency=100):
    return (simulation.data['iterations'] + 1) % frequency == 0


class SaveSimulationData(LogicNode):
    def __init__(self, simulation, directory, save_condition=_save_condition):
        super().__init__(simulation)
        self.save_condition = save_condition
        self.directory = os.path.join(directory,
                                      self.simulation.name_with_timestamp)
        os.makedirs(self.directory)

        self.simulation.field.dump_geometry(
            os.path.join(self.directory, 'geometry.json'))

        self.save_agent_npy = save_npy(self.directory, 'agents')
        self.save_agent_npy.send(None)

        self.save_data_csv = save_csv(self.directory, 'data')
        self.save_data_csv.send(None)

    def update(self):
        save = self.save_condition(self.simulation)

        self.save_agent_npy.send(self.simulation.agents.array)
        self.save_agent_npy.send(save)

        self.save_data_csv.send(self.simulation.data)
        self.save_data_csv.send(save)


def save_simulation_data(simulation, directory):
    node = SaveSimulationData(simulation, directory)
    simulation.logic['Reset'].inject_before(node)


# States

def contains(simulation, vertices, state):
    """Contains

    Args:
        simulation (MultiAgentSimulation):
        vertices (numpy.ndarray): Vertices of a polygon
        state (str):

    Yields:
        int: Number of states that changed
    """
    geom = Path(vertices)
    old_state = simulation.agents.array[state]
    while True:
        position = simulation.agents.array['position']
        new_state = geom.contains_points(position)
        simulation.agents.array[state][:] = new_state
        changed = old_state ^ new_state
        old_state = new_state
        yield np.sum(changed)


class InsideDomain(LogicNode):
    def __init__(self, simulation):
        super().__init__(simulation)
        # TODO: handle domain is None
        self.gen = contains(simulation,
                            np.asarray(self.simulation.field.domain.exterior),
                            'active')

    def update(self, *args, **kwargs):
        self.simulation.data['goal_reached'] += next(self.gen)
