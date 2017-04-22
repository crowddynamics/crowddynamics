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
    * - Obstacles
      - :math:`\mathcal{O}`
      - LineString
    * - Targets
      - :math:`\mathcal{E}` 
      - Linestring
    * - Agents
      - :math:`\mathcal{A}`
      - Circle(s)

"""
import logging
import multiprocessing
import os
from multiprocessing import Process, Event

import numpy as np
from anytree.iterators import PostOrderIter
from loggingtools import log_with
from matplotlib.path import Path

from crowddynamics.config import load_config, MULTIAGENT_CFG, \
    MULTIAGENT_CFG_SPEC
from crowddynamics.core.geometry import geom_to_linear_obstacles
from crowddynamics.core.integrator.integrator import velocity_verlet_integrator
from crowddynamics.core.interactions.interactions import \
    agent_agent_block_list_circular, agent_agent_block_list_three_circle, \
    circular_agent_linear_wall, three_circle_agent_linear_wall
from crowddynamics.core.motion.adjusting import force_adjust_agents, \
    torque_adjust_agents
from crowddynamics.core.motion.fluctuation import force_fluctuation, \
    torque_fluctuation
from crowddynamics.core.steering.navigation import navigation
from crowddynamics.core.steering.navigation import static_potential
from crowddynamics.core.structures.agents import is_model, reset_motion
from crowddynamics.core.vector import angle
from crowddynamics.exceptions import CrowdDynamicsException
from crowddynamics.io import save_data
from crowddynamics.taskgraph import Node

CONFIG = load_config(MULTIAGENT_CFG, MULTIAGENT_CFG_SPEC)


class MultiAgentSimulation(object):
    r"""MultiAgent simulation setup

    1. Set the Field

       1. Domain
       2. Obstacles
       3. Targets (aka exits)

    2. Initialise Agents

       - Set maximum number of agents. This is the limit of the size of array
         inside ``Agent`` class.
       - Select Agent model.

    3. Place Agents into any surface that is contained by the domain.

       - Body type
       - Number of agents that is placed into the surface

    4. Run simulation

    """
    logger = logging.getLogger(__name__)
    # TODO: Properties -> Descriptors
    # https://docs.python.org/3.6/howto/descriptor.html

    def __init__(self):
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
        self.agents_array = None

        # Simulation logic
        self.__tasks = None
        self.iterations = 0

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
    def domain(self, domain):
        """Set simulation domain

        Args:
            domain (Polygon): 
                Subset of real domain :math:`\Omega \subset \mathbb{R}^{2}`.
        """
        self.__domain = domain

    @property
    def obstacles(self):
        """Obstacles"""
        return self.__obstacles

    @obstacles.setter
    def obstacles(self, obstacles):
        """Set obstacles to the simulation

        Args:
            obstacles (MultiLineString): 
        """
        self.__obstacles = obstacles
        self.obstacles_array = geom_to_linear_obstacles(obstacles)

    @property
    def targets(self):
        """Targets"""
        return self.__targets

    @targets.setter
    def targets(self, targets):
        """Set targets to the simulation

        Args:
            targets (BaseGeometry): 
        """
        self.__targets = targets

    @property
    def agents(self):
        """Agent"""
        return self.__agents

    @agents.setter
    def agents(self, agents):
        """Set agents

        Args:
            agent (Agents): 
        """
        self.__agents = agents
        self.agents_array = agents.array

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
            tasks (Node):
        """
        self.__tasks = tasks

    def update(self):
        """Execute new iteration cycle of the simulation."""
        for node in PostOrderIter(self.tasks.root):
            node.update()
        self.iterations += 1


class MultiAgentProcess(Process):
    """MultiAgentProcess

    Class for running MultiAgentSimulation in a new process.
    """
    logger = logging.getLogger(__name__)
    # End of Simulation. Value that is injected into queue when simulation ends.

    class EndProcess(object):
        """Marker for end of simulation"""

    def __init__(self, simulation, queue, maxiterations=None):
        """Init MultiAgentProcess

        Args:
            simulation (MultiAgentSimulation):
            queue (multiprocessing.Queue): 
            maxiterations (int): 
        """
        super(MultiAgentProcess, self).__init__()
        self.simulation = simulation
        self.exit = Event()
        self.maxiter = maxiterations
        self.queue = queue

    @log_with(qualname=True)
    def run(self):
        """Runs simulation process by calling update method repeatedly until
        stop is called. This method is called automatically by Process class
        when start is called."""
        while not self.exit.is_set():
            try:
                self.simulation.update()
            except CrowdDynamicsException as error:
                self.logger.error('Simulation stopped to error: {}'.format(
                    error))
                self.stop()

            if self.maxiter and self.simulation.iterations > self.maxiter:
                self.stop()
        self.queue.put(self.EndProcess)

    @log_with(qualname=True)
    def stop(self):
        """Sets event to true in order to stop the simulation process."""
        self.exit.set()


@log_with()
def run_sequentially(*simulations, maxiter=None):
    """Run new simulation sequentially

    Args:
        simulation (MultiAgentSimulation):
    """
    for simulation in simulations:
        iterations = 0
        conds = True
        while conds:
            simulation.update()
            iterations += 1
            if maxiter:
                conds = iterations <= maxiter


@log_with()
def run_parallel(*simulations, queue, maxiter=None):
    """Run multiagent simulations as a new process

    Wraps MultiAgentSimulations in MultiAgentProcess class and starts them.
    Started process is yielded and can be stopped with stop method. When
    stop is called process will put ``MultiAgentProcess.EOS`` into the queue as
    sign of final value.

    Args:
        simulations (MultiAgentSimulation):
            Iterable of multiagent simulations to be run in a new process.

    Yields:
        MultiAgentProcess: Started simulation process
    """
    for simulation in simulations:
        process = MultiAgentProcess(simulation, queue, maxiter)
        process.start()
        yield process


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
        self.logger.info('Init MASNode: {}'.format(self.__class__.__name__))


class Integrator(MASNode):
    r"""Integrator"""

    def __init__(self, simulation):
        super().__init__(simulation)
        self.dt_min = CONFIG['integrator']['dt_min']
        self.dt_max = CONFIG['integrator']['dt_max']
        self.time_tot = np.float64(0.0)
        self.dt_prev = np.float64(np.nan)

    def update(self):
        # self.dt_prev = euler_integrator(self.simulation.agents_array,
        #                                 self.dt_min, self.dt_max)
        self.dt_prev = velocity_verlet_integrator(self.simulation.agents_array,
                                                  self.dt_min, self.dt_max)
        self.time_tot += self.dt_prev


class Fluctuation(MASNode):
    r"""Fluctuation"""

    def update(self):
        agent = self.simulation.agents_array
        agent['force'] += force_fluctuation(agent['mass'],
                                            agent['std_rand_force'])
        if is_model(self.simulation.agents_array, 'three_circle'):
            agent['torque'] += torque_fluctuation(agent['inertia_rot'],
                                                  agent['std_rand_torque'])


class Adjusting(MASNode):
    r"""Adjusting"""

    def update(self):
        agents = self.simulation.agents_array
        force_adjust_agents(agents)
        if is_model(self.simulation.agents_array, 'three_circle'):
            torque_adjust_agents(agents)


class AgentAgentInteractions(MASNode):
    r"""AgentAgentInteractions"""

    def update(self):
        if is_model(self.simulation.agents_array, 'circular'):
            agent_agent_block_list_circular(self.simulation.agents_array)
        elif is_model(self.simulation.agents_array, 'three_circle'):
            agent_agent_block_list_three_circle(self.simulation.agents_array)


class AgentObstacleInteractions(MASNode):
    r"""AgentObstacleInteractions"""

    def update(self):
        if is_model(self.simulation.agents_array, 'circular'):
            circular_agent_linear_wall(self.simulation.agents_array,
                                       self.simulation.obstacles_array)
        elif is_model(self.simulation.agents_array, 'three_circle'):
            three_circle_agent_linear_wall(self.simulation.agents_array,
                                           self.simulation.obstacles_array)


class Navigation(MASNode):
    r"""Handles navigation in multi-agent simulation."""

    def __init__(self, simulation):
        super().__init__(simulation)
        self.mgrid, self.distance_map, self.direction_map = static_potential(
            self.simulation.domain, self.simulation.targets,
            self.simulation.obstacles, CONFIG['navigation']['step'],
            radius=CONFIG['navigation']['radius'],
            value=CONFIG['navigation']['value'])

    def update(self):
        position = self.simulation.agents_array['position']
        direction = self.simulation.agents_array['target_direction']
        d = navigation(position, direction, self.mgrid, self.direction_map)
        self.simulation.agents_array['target_direction'][:] = d


class Orientation(MASNode):
    r"""Target orientation"""

    def update(self):
        if is_model(self.simulation.agents_array, 'three_circle'):
            dir_to_orient = angle(self.simulation.agents_array['target_direction'])
            self.simulation.agents_array['target_orientation'] = dir_to_orient


class ExitSelection(MASNode):
    """Exit selection policy."""

    def update(self):
        pass


class Reset(MASNode):
    r"""Reset"""

    def update(self):
        reset_motion(self.simulation.agents_array)
        # TODO: reset agent neighbor


class SaveAgentsData(MASNode):
    r"""Saves data to hdf5 file."""

    def __init__(self, simulation, directory):
        super().__init__(simulation)
        self.save_data = save_data(directory, 'agents')
        self.save_data.send(None)
        self.iterations = 0

    def update(self, frequency=100):
        self.save_data.send(self.simulation.agents_array)
        self.save_data.send(self.iterations % frequency == 0)
        self.iterations += 1


class Contains(MASNode):
    """Contains"""

    def __init__(self, simulation, polygon):
        super().__init__(simulation)

        self.path = Path(np.asarray(polygon.exterior))
        self.inside = np.zeros(self.simulation.agents.size, np.bool8)
        self.update()

    def update(self, *args, **kwargs):
        position = self.simulation.agents_array['position']
        inside = self.path.contains_points(position)
        # out: True  -> False
        # in:  False -> True
        changed = self.inside ^ inside
        self.inside = inside
        diff = np.sum(changed)
