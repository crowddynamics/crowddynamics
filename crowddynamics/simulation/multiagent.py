"""Multi-Agent Simulation

Tools for creating multiagent simulations.

.. csv-table:: Model Specification

   "Continuous Space", ":math:`\mathbb{R}^2`"
   "Microscopic", "Agents are modelled as rigid bodies."
   "Social Force Model", "Classical mechanics for modelling movement."

Multi-agent simulation attributes

.. list-table::
    :header-rows: 1

    * - Name
      - Symbol
      - Type
    * - Domain
      - :math:`\Omega`
      - Polygon
    * - BaseGeometry
      - :math:`\mathcal{O}`
      - BaseGeometry
    * - Targets
      - :math:`\mathcal{E}` 
      - BaseGeometry
    * - Agents
      - :math:`\mathcal{A}`
      - Circle(s)

"""
import json
import logging
import multiprocessing
import os
from collections import OrderedDict
from multiprocessing import Process, Event
from typing import Optional
from datetime import datetime

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
from crowddynamics.core.integrator.integrator import velocity_verlet_integrator
from crowddynamics.core.interactions.interactions import \
    circular_agent_linear_wall, three_circle_agent_linear_wall, \
    agent_agent_block_list
from crowddynamics.core.motion.adjusting import force_adjust_agents, \
    torque_adjust_agents
from crowddynamics.core.motion.fluctuation import force_fluctuation, \
    torque_fluctuation
from crowddynamics.core.steering.navigation import navigation
from crowddynamics.core.steering.navigation import static_potential
from crowddynamics.core.structures.agents import is_model, reset_motion, Agents
from crowddynamics.core.vector import angle
from crowddynamics.exceptions import CrowdDynamicsException, InvalidType, \
    InvalidValue
from crowddynamics.io import save_npy, save_csv
from crowddynamics.taskgraph import Node


def validate_geom(geom, name, types, optional=False):
    """Validate simulation geometry

    Args:
        geom (BaseGeometry): 
        name (str): 
        types: Type of iterable of types. 
        optional (bool): True if None is valid type else False.
    """
    is_optional = geom is None if optional else False

    if not isinstance(geom, types) and not is_optional:
        raise InvalidType('{name} should be instance of {types}.'.format(
            name=name.capitalize(), types=types))

    if geom.is_empty or not geom.is_valid or not geom.is_simple:
        raise InvalidValue('{name} should not be empty and should be valid '
                           'and simple.'.format(name=name.capitalize()))


class MultiAgentSimulation(object):
    """Multi-Agent Simulation class"""

    def __init__(self, configfile=MULTIAGENT_CFG):
        self.configs = load_config(configfile, MULTIAGENT_CFG_SPEC)

        # Name
        self.__name = None

        # Geometry
        self.__domain = None
        self.__obstacles = None
        self.__targets = None
        self.__agents = None

        # Numerical types for geometry
        self.domain_array = None
        self.obstacles_array = None
        self.targets_array = None

        # Simulation logic
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
        """Set name for the simulations

        Args:
            name (str): 
        """
        self.__name = name

    @property
    def domain(self):
        """Set domain"""
        return self.__domain

    @domain.setter
    def domain(self, domain: Optional[Polygon]):
        """Set simulation domain

        Args:
            domain (Polygon): 
                Subset of real domain :math:`\Omega \subset \mathbb{R}^{2}`.
        """
        validate_geom(domain, 'domain', Polygon, True)
        self.__domain = domain

    @property
    def obstacles(self):
        """Obstacles"""
        return self.__obstacles

    @obstacles.setter
    def obstacles(self, obstacles: Optional[BaseGeometry]):
        """Set obstacles to the simulation

        Args:
            obstacles (MultiLineString): 
        """
        validate_geom(obstacles, 'obstacles', BaseGeometry, True)
        self.__obstacles = obstacles
        self.obstacles_array = geom_to_linear_obstacles(obstacles)

    @property
    def targets(self):
        """Targets"""
        return self.__targets

    @targets.setter
    def targets(self, targets: Optional[BaseGeometry]):
        """Set targets to the simulation

        Args:
            targets (BaseGeometry): 
        """
        validate_geom(targets, 'targets', BaseGeometry, True)
        self.__targets = targets

    @property
    def agents(self):
        """Agent"""
        return self.__agents

    @agents.setter
    def agents(self, agents: Agents):
        """Set agents

        Args:
            agent (Agents): 
        """
        if isinstance(agents, Agents):
            self.__agents = agents
        else:
            raise InvalidType('Agents should be type {}'.format(Agents))

    @property
    def tasks(self):
        """Tasks

        Returns:
            Node: 
        """
        return self.__tasks

    @tasks.setter
    def tasks(self, tasks):
        """Set task graph to the simulation

        Args:
            tasks (MASNode):
        """
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

    def dump_geometry(self, fname):
        """Dump simulation geometry into JSON"""
        def _mapping(geom):
            if geom is None:
                return None
            else:
                return mapping(geom)

        _, ext = os.path.splitext(fname)
        if ext != '.json':
            fname += '.json'

        with open(fname, 'w') as fp:
            obj = {
                'domain': _mapping(self.domain),
                'obstacles': _mapping(self.obstacles),
                'targets': _mapping(self.targets),
            }
            # TODO: maybe compact layout?
            json.dump(obj, fp, indent=2, separators=(', ', ': '))

    def load_geometry(self, fname):
        """Load simulation geometry from JSON"""
        def _shape(geom):
            if geom is None:
                return None
            else:
                return shape(geom)

        with open(fname, 'r') as fp:
            obj = json.load(fp)
            self.domain = _shape(obj['domain'])
            self.obstacles = _shape(obj['obstacles'])
            self.targets = _shape(obj['targets'])

    def dump_config(self):
        raise NotImplementedError


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


class MASNode(Node):
    # TODO: log_with, __init__, update
    logger = logging.getLogger(__name__)

    def __init__(self, simulation):
        """MultiAgentSimulation TaskNode
        
        Args:
            simulation (MultiAgentSimulation): 
        """
        super(MASNode, self).__init__()
        assert isinstance(simulation, MultiAgentSimulation)
        self.simulation = simulation
        self.configs = self.simulation.configs
        self.data = self.simulation.data
        self.logger.info('Init MASNode: {}'.format(self.__class__.__name__))


class Integrator(MASNode):
    r"""Integrator"""

    def __init__(self, simulation):
        super().__init__(simulation)
        self.dt_min = self.configs['integrator']['dt_min']
        self.dt_max = self.configs['integrator']['dt_max']

    def update(self):
        self.data['dt'] = velocity_verlet_integrator(
            self.simulation.agents.array,
            self.dt_min, self.dt_max)
        self.data['time_tot'] += self.data['dt']


class Fluctuation(MASNode):
    r"""Fluctuation"""

    def update(self):
        agent = self.simulation.agents.array
        agent['force'] += force_fluctuation(agent['mass'],
                                            agent['std_rand_force'])
        if is_model(self.simulation.agents.array, 'three_circle'):
            agent['torque'] += torque_fluctuation(agent['inertia_rot'],
                                                  agent['std_rand_torque'])


class Adjusting(MASNode):
    r"""Adjusting"""

    def update(self):
        agents = self.simulation.agents.array
        force_adjust_agents(agents)
        if is_model(self.simulation.agents.array, 'three_circle'):
            torque_adjust_agents(agents)


class AgentAgentInteractions(MASNode):
    r"""AgentAgentInteractions"""

    def update(self):
        agent_agent_block_list(self.simulation.agents.array)


class AgentObstacleInteractions(MASNode):
    r"""AgentObstacleInteractions"""

    def update(self):
        if is_model(self.simulation.agents.array, 'circular'):
            circular_agent_linear_wall(self.simulation.agents.array,
                                       self.simulation.obstacles_array)
        elif is_model(self.simulation.agents.array, 'three_circle'):
            three_circle_agent_linear_wall(self.simulation.agents.array,
                                           self.simulation.obstacles_array)


class Navigation(MASNode):
    r"""Handles navigation in multi-agent simulation."""

    def __init__(self, simulation):
        super().__init__(simulation)
        self.mgrid, self.distance_map, self.direction_map = static_potential(
            self.simulation.domain, self.simulation.targets,
            self.simulation.obstacles, self.configs['navigation']['step'],
            radius=self.configs['navigation']['radius'],
            value=self.configs['navigation']['value'])

    def update(self):
        position = self.simulation.agents.array['position']
        direction = self.simulation.agents.array['target_direction']
        d = navigation(position, direction, self.mgrid, self.direction_map)
        self.simulation.agents.array['target_direction'][:] = d


class Orientation(MASNode):
    r"""Target orientation"""

    def update(self):
        if is_model(self.simulation.agents.array, 'three_circle'):
            dir_to_orient = angle(
                self.simulation.agents.array['target_direction'])
            self.simulation.agents.array['target_orientation'] = dir_to_orient


class ExitSelection(MASNode):
    """Exit selection policy."""

    def update(self):
        pass


class Reset(MASNode):
    r"""Reset"""

    def update(self):
        reset_motion(self.simulation.agents.array)
        # TODO: reset agent neighbor


def _save_condition(simulation, frequency=100):
    return (simulation.data['iterations'] + 1) % frequency == 0


class SaveAgentsData(MASNode):
    r"""Saves data to .npy file."""

    def __init__(self, simulation, directory, save_condition=_save_condition):
        super().__init__(simulation)
        self.save_condition = save_condition
        self.directory = os.path.join(directory, self.simulation.stamped_name())
        os.makedirs(self.directory)

        self.simulation.dump_geometry(os.path.join(self.directory, 'geometry.json'))

        self.save_agent_npy = save_npy(self.directory, 'agents')
        self.save_agent_npy.send(None)

        self.save_data_csv = save_csv(self.directory, 'data')
        self.save_data_csv.send(None)

    def update(self):
        save = self.save_condition(self.simulation)

        self.save_agent_npy.send(self.simulation.agents.array)
        self.save_agent_npy.send(save)

        self.save_data_csv.send(self.data)
        self.save_data_csv.send(save)


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


class InsideDomain(MASNode):
    """Contains"""

    def __init__(self, simulation):
        super().__init__(simulation)
        # TODO: handle domain is None
        self.gen = contains(simulation,
                            np.asarray(self.simulation.domain.exterior),
                            'active')

    def update(self, *args, **kwargs):
        self.data['goal_reached'] += next(self.gen)
