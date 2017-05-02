r"""Multi-Agent Simulation

Tools for creating multiagent simulations.

.. csv-table:: Model Specification

   "Continuous Space", ":math:`\mathbb{R}^2`"
   "Microscopic", "Agents are modelled as rigid bodies."
   "Social Force Model", "Classical mechanics for modelling movement."

Social force model

Dirk Helbing, a pioneer of social force model, describes social forces

   *These forces are not directly exerted by the pedestrians’ personal 
    environment, but they are a measure for the internal motivations of the 
    individuals to perform certain actions (movements).*
    
    -- Dirk Helbing

Total force exerted on the agent

.. math::
   \mathbf{f}_{i}(t) = \mathbf{f}_{i}^{adj} + \sum_{j\neq i}^{} 
   \left(\mathbf{f}_{ij}^{soc} + \mathbf{f}_{ij}^{c}\right) + 
   \sum_{w}^{} \mathbf{f}_{iw}^{c},

Total torque on the agent

.. math::
   M_{i}(t) = M_{i}^{adj} + \sum_{j\neq i}^{} 
   \left(M_{ij}^{soc} + M_{ij}^{c}\right) + 
   \sum_{w}^{} M_{iw}^{c},

In our model social forces between agents and obstacles are handled by adjusting
force and torque.
"""
import json
import logging
import multiprocessing
import os
from collections import OrderedDict, Iterable
from datetime import datetime
from functools import reduce, lru_cache
from itertools import chain
from multiprocessing import Process, Event
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from anytree.iterators import PostOrderIter
from loggingtools import log_with
from matplotlib.path import Path
from shapely.geometry import Polygon
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from crowddynamics.config import load_config, MULTIAGENT_CFG, \
    MULTIAGENT_CFG_SPEC
from crowddynamics.core.geometry import geom_to_linear_obstacles
from crowddynamics.core.integrator import velocity_verlet_integrator
from crowddynamics.core.interactions.interactions import \
    agent_agent_block_list, agent_obstacle
from crowddynamics.core.motion.adjusting import force_adjust_agents, \
    torque_adjust_agents
from crowddynamics.core.motion.fluctuation import force_fluctuation, \
    torque_fluctuation
from crowddynamics.core.random.sampling import polygon_sample
from crowddynamics.core.steering.navigation import navigation, static_potential
from crowddynamics.core.steering.orientation import \
    orient_towards_target_direction
from crowddynamics.core.structures.agents import is_model, reset_motion, Agents
from crowddynamics.core.tree import Node
from crowddynamics.exceptions import CrowdDynamicsException, InvalidType, \
    InvalidValue
from crowddynamics.io import save_npy, save_csv
from crowddynamics.visualization import plot_navigation

# Types
Domain = Optional[Polygon]
Obstacles = Optional[BaseGeometry]
Spawn = Polygon
Targets = BaseGeometry


def validate_geom(geom, name, types, optional=False):
    """Validate simulation geometry

    Args:
        geom (BaseGeometry): 
        name (str): 
        types: Type of iterable of types. 
        optional (bool): True if None is valid type else False.
    """
    if optional and geom is None:
        return

    if not isinstance(geom, types):
        raise InvalidType('{name} should be instance of {types}.'.format(
            name=name.capitalize(), types=types))

    if geom.is_empty or not geom.is_valid or not geom.is_simple:
        raise InvalidValue('{name} should not be empty and should be valid '
                           'and simple.'.format(name=name.capitalize()))


def samples(spawn, obstacles, radius=0.3):
    """Generates positions for agents"""
    geom = spawn - obstacles.buffer(radius) if obstacles else spawn
    vertices = np.asarray(geom.exterior)
    return polygon_sample(vertices)


def union(*geoms):
    """Union of geometries"""
    return reduce(lambda x, y: x | y, geoms)


class Field(object):
    """Class for settings simulation geometry

    - domain :math:`\Omega \subset \mathbb{R}^{2}`
    - obstacles :math:`\mathcal{O}`
    - targets :math:`\mathcal{E}`
    - spawns :math:`\mathcal{S}`
    """

    def __init__(self, name=None, domain=None, obstacles=None):
        # TODO: invalidate caches if field changes?

        # Internal data
        self.__name = None
        self.__domain = None
        self.__obstacles = None
        self.__targets = []
        self.__spawns = []

        # Uses setters to set properties
        self.name = name
        self.domain = domain
        self.obstacles = obstacles

    @property
    def name(self):
        """Name of the simulation. Defaults to __class__.__name__."""
        return self.__name if self.__name else self.__class__.__name__

    @name.setter
    def name(self, name: str):
        """Set name for the field"""
        self.__name = name

    @property
    def domain(self):
        """Domain"""
        return self.__domain

    @property
    def obstacles(self):
        """Obstacles"""
        return self.__obstacles

    @property
    def spawns(self):
        """Spawns"""
        return self.__spawns

    @property
    def targets(self):
        """Targets"""
        return self.__targets

    @domain.setter
    def domain(self, domain: Domain):
        validate_geom(domain, 'domain', Polygon, True)
        self.__domain = domain

    @obstacles.setter
    def obstacles(self, obstacles: Obstacles):
        validate_geom(obstacles, 'obstacles', (BaseGeometry,), True)
        self.__obstacles = obstacles

    def set_domain_convex_hull(self):
        """Set domain from the convex hull of union of all other objects"""
        self.domain = (self.obstacles | union(*self.targets) |
                       union(*self.spawns)).convex_hull

    def add_spawns(self, *spawns: Spawn):
        """Add new spawns"""
        for spawn in spawns:
            # TODO: spawn should be convex
            validate_geom(spawn, 'spawn', (Polygon,), False)
            self.__spawns.append(spawn)

    def add_targets(self, *targets: Targets):
        """Add new targets"""
        for target in targets:
            validate_geom(target, 'target', (BaseGeometry,), False)
            self.__targets.append(target)

    def sample_spawn(self, spawn_index: int, radius: float=0.3):
        """Generator for sampling points inside spawn without overlapping with
        obstacles"""
        return samples(self.spawns[spawn_index], self.obstacles, radius)

    @lru_cache()
    def navigation_to_target(self, index, step, radius, value):
        if not self.targets:
            raise CrowdDynamicsException('No targets are set.')
        if isinstance(index, (int, np.uint8)):
            target = self.targets[index]
        elif index == 'closest':
            target = union(*self.targets)
        else:
            raise InvalidType('Index "{0}" should be integer or '
                              '"closest".'.format(index))

        return static_potential(self.domain, target, self.obstacles, step,
                                radius, value)

    def dump_json(self, fname: str):
        """Dump field into JSON"""
        def _mapping(geom):
            if isinstance(geom, Iterable):
                return list(map(mapping, geom))
            else:
                return mapping(geom) if geom else geom

        _, ext = os.path.splitext(fname)
        if ext != '.json':
            fname += '.json'

        with open(fname, 'w') as fp:
            obj = {
                'domain': _mapping(self.domain),
                'obstacles': _mapping(self.obstacles),
                'targets': _mapping(self.targets),
                'spawns': _mapping(self.spawns)
            }
            # TODO: maybe compact layout?
            json.dump(obj, fp, indent=2, separators=(', ', ': '))

    def load_json(self, fname: str):
        """Load field from JSON"""
        def _shape(geom):
            if isinstance(geom, Iterable):
                return list(map(shape, geom))
            else:
                return shape(geom) if geom else geom

        with open(fname, 'r') as fp:
            obj = json.load(fp)
            self.domain = _shape(obj['domain'])
            self.obstacles = _shape(obj['obstacles'])
            self.add_targets(*_shape(obj['targets']))
            self.add_spawns(*_shape(obj['spawns']))

    def plot_navigation(self, step=0.02, radius=0.3, value=0.3):
        indices = chain(range(len(self.targets)), ('closest',))
        for index in indices:
            fname = '{}_navigation_{}.pdf'.format(self.name, index)
            mgrid, distance_map, direction_map = \
                self.navigation_to_target(index, step, radius, value)

            fig, ax = plt.subplots(figsize=(12, 12))
            plot_navigation(fig, ax,
                            mgrid.values,
                            distance_map,
                            direction_map,
                            frequency=5
                            )
            # plt.savefig(fname)
            plt.show()


class MultiAgentSimulation(object):
    """Multi-Agent Simulation class"""

    def __init__(self, name=None, configfile=MULTIAGENT_CFG):
        self.configs = load_config(configfile, MULTIAGENT_CFG_SPEC)

        self.__name = name
        self.__field = None
        self.__agents = None
        self.__tasks = None

        # Generated simulation data that is shared between task nodes.
        # This should be data that can be updated and should be saved on
        # every iteration.
        self.data = OrderedDict()
        self.data['iterations'] = 0
        self.data['time_tot'] = 0.0
        self.data['dt'] = 0.0
        self.data['goal_reached'] = 0

    @property
    def name(self):
        """Name of the simulation. Defaults to __class__.__name__."""
        return self.__name if self.__name else self.__class__.__name__

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def field(self):
        """Field (geometry) of the simulation."""
        return self.__field

    @field.setter
    def field(self, field: Field):
        self.__field = field

    @property
    def agents(self):
        """Agents"""
        return self.__agents

    @agents.setter
    def agents(self, agents: Agents):
        if isinstance(agents, Agents):
            self.__agents = agents
        else:
            raise InvalidType('Agents should be type {}'.format(Agents))

    @property
    def tasks(self):
        """Logic of the simulation"""
        return self.__tasks

    @tasks.setter
    def tasks(self, tasks):
        self.__tasks = tasks

    def update(self):
        """Execute new iteration cycle of the simulation."""
        for node in PostOrderIter(self.tasks.root):
            node.update()
        self.data['iterations'] += 1

    @log_with(qualname=True, ignore={'self'})
    def run(self, exit_condition=lambda _: False):
        """Run simulation until exit condition returns True.

        Args:
            exit_condition (Callable[MultiAgentSimulation, bool]): 
        """
        while not exit_condition(self):
            self.update()

    def stamped_name(self):
        """Timestamped name"""
        return self.name + '_' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')

    def dump_config(self):
        raise NotImplementedError

    def plot_tasks(self, fname):
        from anytree.dotexport import RenderTreeGraph
        name, ext = os.path.splitext(fname)
        if ext != '.png':
            fname += '.png'
        RenderTreeGraph(self.tasks.root).to_picture(fname)


class MultiAgentProcess(Process):
    """Class for running MultiAgentSimulation in a new process."""
    logger = logging.getLogger(__name__)

    class EndProcess(object):
        """Marker for end of simulation"""

    def __init__(self, simulation, queue):
        """MultiAgentProcess

        Examples:
            >>> process = MultiAgentProcess(simulation, queue)
            >>> process.start()  # Starts the simulation
            >>> ...
            >>> process.stop()  # Stops the simulation

        Args:
            simulation (MultiAgentSimulation):
            queue (multiprocessing.Queue): 
        """
        super(MultiAgentProcess, self).__init__()
        self.simulation = simulation
        self.exit = Event()
        self.queue = queue

    @log_with(qualname=True)
    def run(self):
        """Runs simulation process by calling update method repeatedly until
        stop is called. This method is called automatically by Process class
        when start is called."""
        try:
            self.simulation.run(exit_condition=lambda _: self.exit.is_set())
        except CrowdDynamicsException as error:
            self.logger.error(
                'Simulation stopped to error: {}'.format(error))
            self.stop()
        self.queue.put(self.EndProcess)

    @log_with(qualname=True)
    def stop(self):
        """Sets event to true in order to stop the simulation process."""
        self.exit.set()


# Simulation Logic


class LogicNode(Node):
    # logger = logging.getLogger(__name__)

    def __init__(self, simulation: MultiAgentSimulation):
        """MultiAgentSimulation Logic Node"""
        super(LogicNode, self).__init__()
        self.simulation = simulation


class Integrator(LogicNode):
    def update(self):
        dt_min = self.simulation.configs['integrator']['dt_min']
        dt_max = self.simulation.configs['integrator']['dt_max']
        dt = velocity_verlet_integrator(self.simulation.agents.array,
                                        dt_min, dt_max)
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
    def update(self):
        agent_agent_block_list(self.simulation.agents.array)


class AgentObstacleInteractions(LogicNode):
    def update(self):
        agents = self.simulation.agents.array
        obstacles = geom_to_linear_obstacles(self.simulation.field.obstacles)
        agent_obstacle(agents, obstacles)


class Navigation(LogicNode):
    def update(self):
        step = self.simulation.configs['navigation']['step']
        radius = radius = self.simulation.configs['navigation']['radius']
        value = value = self.simulation.configs['navigation']['value']
        agents = self.simulation.agents.array
        targets = agents['target']
        for target in set(targets):
            mgrid, distance_map, direction_map = \
                self.simulation.field.navigation_to_target(
                    target, step, radius, value)
            navigation(agents, targets == target, mgrid, direction_map)


class Orientation(LogicNode):
    def update(self):
        if is_model(self.simulation.agents.array, 'three_circle'):
            orient_towards_target_direction(self.simulation.agents.array)


class Reset(LogicNode):
    def update(self):
        reset_motion(self.simulation.agents.array)
        # TODO: reset agent neighbor


def _save_condition(simulation, frequency=100):
    return (simulation.data['iterations'] + 1) % frequency == 0


class SaveSimulationData(LogicNode):
    def __init__(self, simulation, directory, save_condition=_save_condition):
        super().__init__(simulation)
        self.save_condition = save_condition
        self.directory = os.path.join(directory, self.simulation.stamped_name())
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
    simulation.tasks['Reset'].inject_before(node)


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
