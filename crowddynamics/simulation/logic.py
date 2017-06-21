import os
from collections import Callable

import numpy as np
from loggingtools.log_with import log_with
from matplotlib.path import Path
from traitlets.traitlets import Float, Instance, Unicode, default, \
    Int

from crowddynamics.core.evacuation import exit_detection
from crowddynamics.core.geometry import geom_to_linear_obstacles
from crowddynamics.core.integrator import velocity_verlet_integrator
from crowddynamics.core.interactions.interactions import agent_agent_block_list, \
    agent_obstacle
from crowddynamics.core.motion.adjusting import force_adjust_agents, \
    torque_adjust_agents
from crowddynamics.core.motion.fluctuation import force_fluctuation, \
    torque_fluctuation
from crowddynamics.core.steering.collective_motion import herding_block_list
from crowddynamics.core.steering.navigation import getdefault
from crowddynamics.core.steering.obstacle_handling import \
    obstacle_handling_continuous
from crowddynamics.core.steering.orientation import \
    orient_towards_target_direction
from crowddynamics.core.structures import obstacle_type_linear
from crowddynamics.io import save_npy, save_csv, save_geometry_json
from crowddynamics.simulation.agents import is_model, NO_TARGET
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

    def __init__(self, simulation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation = simulation

    def update(self):
        raise NotImplementedError


# Motion

class Reset(LogicNode):
    def update(self):
        agents = self.simulation.agents.array
        agents[:]['force'] = 0
        if is_model(agents, 'three_circle'):
            agents[:]['torque'] = 0


class Integrator(LogicNode):
    dt_min = Float(default_value=0.01, min=0, help='Minimum timestep')
    dt_max = Float(default_value=0.01, min=0, help='Maximum timestep')

    def update(self):
        dt = velocity_verlet_integrator(
            self.simulation.agents.array, self.dt_min, self.dt_max)
        self.simulation.data['dt'] = dt
        self.simulation.data['time_tot'] += dt


class Fluctuation(LogicNode):
    def update(self):
        agents = self.simulation.agents.array
        force = force_fluctuation(agents['mass'], agents['std_rand_force'])
        agents['force'] += force
        if is_model(agents, 'three_circle'):
            torque = torque_fluctuation(agents['inertia_rot'],
                                        agents['std_rand_torque'])
            agents['torque'] += torque


class Adjusting(LogicNode):
    def update(self):
        agents = self.simulation.agents.array
        force_adjust_agents(agents)
        if is_model(agents, 'three_circle'):
            torque_adjust_agents(agents)


class AgentAgentInteractions(LogicNode):
    sight_soc = Float(
        default_value=3.0,
        min=0,
        help='')
    max_agent_radius = Float(
        default_value=0.3,
        min=0,
        help='')
    f_soc_max = Float(
        default_value=2e3,
        min=0,
        help='')
    cell_size = Float(
        min=0,
        help='')

    @default('cell_size')
    def _default_cell_size(self):
        return self.sight_soc + 2 * self.max_agent_radius

    def update(self):
        agent_agent_block_list(self.simulation.agents.array, self.cell_size)


class AgentObstacleInteractions(LogicNode):
    def update(self):
        agents = self.simulation.agents.array
        if self.simulation.field.obstacles is None:
            obstacles = np.zeros(shape=0, dtype=obstacle_type_linear)
        else:
            obstacles = geom_to_linear_obstacles(self.simulation.field.obstacles)
        agent_obstacle(agents, obstacles)


# Steering

class Navigation(LogicNode):
    step = Float(
        default_value=0.1,
        min=0,
        help='Step size for meshgrid used for discretization.')
    radius = Float(
        default_value=0.5,
        min=0,
        help='')
    strength = Float(
        default_value=0.3,
        min=0, max=1,
        help='')

    def update(self):
        agents = self.simulation.agents.array
        targets = set(range(len(self.simulation.field.targets)))
        for target in targets:
            if target == NO_TARGET:
                continue

            mask = agents['target'] == target
            if len(mask) == 0:
                continue

            mgrid, distance_map, direction_map = \
                self.simulation.field.navigation_to_target(
                    target, self.step, self.radius, self.strength)

            # Navigation
            # Flip x and y to array index i and j
            indices = np.fliplr(mgrid.indicer(agents[mask]['position']))
            new_direction = getdefault(indices, direction_map,
                                       agents[mask]['target_direction'])
            agents['target_direction'][mask] = new_direction


class LeaderFollower(LogicNode):
    sight_follower = Float(
        default_value=20.0,
        min=0,
        help='Maximum distance between agents that are accounted as neighbours '
             'that can be followed.')
    size_nearest_leaders = Int(
        default_value=1,
        min=0,
        help='')
    size_nearest_other = Int(
        default_value=5,
        min=0,
        help='Maximum number of nearest agents inside sight_herding radius '
             'that herding agent are following.')

    step = Float(
        default_value=0.1,
        min=0,
        help='Step size for meshgrid used for discretization.')
    radius = Float(
        default_value=0.5,
        min=0,
        help='')
    strength = Float(
        default_value=0.3,
        min=0, max=1,
        help='')

    def update(self):
        agents = self.simulation.agents.array
        if self.simulation.field.obstacles is None:
            obstacles = np.zeros(shape=0, dtype=obstacle_type_linear)
        else:
            obstacles = geom_to_linear_obstacles(self.simulation.field.obstacles)

        is_herding = agents['is_herding']
        direction_herding = herding_block_list(
            agents['position'], agents['velocity'],
            is_herding, agents['is_leader'],
            self.sight_follower, self.size_nearest_leaders,
            self.size_nearest_other, obstacles)

        # agents['target_direction'][is_herding] = direction_herding[is_herding]

        # TODO: add obstacle avoidance
        mgrid = self.simulation.field.meshgrid(self.step)
        dir_map_obs, dmap_obs = self.simulation.field.direction_map_obstacles(self.step)
        indices = np.fliplr(mgrid.indicer(agents['position'][is_herding]))

        direction = obstacle_handling_continuous(
            dmap_obs, dir_map_obs, direction_herding[is_herding], indices,
            self.radius, self.strength)

        agents['target_direction'][is_herding] = direction


class ExitDetection(LogicNode):
    """Herding agents can detect an exit that is within exit detection range"""
    detection_range = Float(
        default_value=20.0,
        min=1.0)

    def update(self):
        agents = self.simulation.agents.array
        if self.simulation.field.obstacles is None:
            obstacles = np.zeros(shape=0, dtype=obstacle_type_linear)
        else:
            obstacles = geom_to_linear_obstacles(self.simulation.field.obstacles)
        targets = self.simulation.field.targets

        # Assume here that target
        # TODO: check that this is correct
        center_door = np.stack(
            [np.mean(np.asarray(target), axis=0) for target in targets])

        is_herding = agents['is_herding']
        position = agents['position'][is_herding]

        targets = exit_detection(center_door, position, obstacles,
                                 self.detection_range)
        agents['target'][is_herding] = targets
        # Not herding anymore
        agents['is_herding'][is_herding] = targets == NO_TARGET


class Orientation(LogicNode):
    def update(self):
        if is_model(self.simulation.agents.array, 'three_circle'):
            orient_towards_target_direction(self.simulation.agents.array)


# IO

class SaveSimulationData(LogicNode):
    """Logic for saving simulation data.

    Saved once
    - Geometry
    - Metadata

    Saved continuously
    - Agents
    - Data

    Examples:
        >>> def save_condition(simulation, frequency=100):
        >>>     return (simulation.data['iterations'] + 1) % frequency == 0
        >>>
        >>> node = SaveSimulationData(save_condition=save_condition,
        >>>                           base_directory='.')
    """
    save_condition = Instance(
        Callable,
        # allow_none=True,
        help='Function to trigger saving of data.')
    base_directory = Unicode(
        default_value='.',
        help='Path to the directory where simulation data should be saved.')
    save_directory = Unicode(
        help='Name of the directory to save current simulation.')

    def __init__(self, simulation, *args, **kwargs):
        super().__init__(simulation, *args, **kwargs)
        os.makedirs(self.full_path, exist_ok=True)

        # Metadata
        save_data_csv = save_csv(self.full_path, 'metadata')
        save_data_csv.send(None)
        save_data_csv.send(self.simulation.metadata)
        save_data_csv.send(True)

        # Geometry
        geometries = {name: getattr(self.simulation.field, name) for name in
                      ('domain', 'obstacles', 'targets', 'spawns')}
        save_geometry_json(os.path.join(self.full_path, 'geometry.json'),
                           geometries)

        # Data
        self.save_data_csv = save_csv(self.full_path, 'data')
        self.save_data_csv.send(None)

        # Agents
        self.save_agent_npy = save_npy(self.full_path, 'agents')
        self.save_agent_npy.send(None)

    @property
    def full_path(self):
        return os.path.join(os.path.abspath(self.base_directory),
                            self.save_directory)

    @default('save_directory')
    def _default_save_directory(self):
        return self.simulation.name_with_timestamp

    # @validate('save_directory')
    # def _valid_save_directory(self, proposal):
    #     path = proposal['value']
    #     new = os.path.join(self.base_directory, path)
    #     if os.path.exists(new):
    #         raise ValidationError('Path: "{}" already exists.'.format(new))
    #     return path

    # @observe('save_directory')
    # def _observe_save_directory(self, change):
    #     os.makedirs(self.full_path, exist_ok=True)

    def add_to_simulation_logic(self):
        self.simulation.logic['Reset'].inject_before(self)

    @log_with(timed=True, arguments=False)
    def update(self):
        save = self.save_condition(self.simulation)

        self.save_agent_npy.send(self.simulation.agents.array)
        self.save_agent_npy.send(save)

        self.save_data_csv.send(self.simulation.data)
        self.save_data_csv.send(save)


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

    def update(self):
        self.simulation.data['goal_reached'] += next(self.gen)
