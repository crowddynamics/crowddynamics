"""Tools for creating multiagent simulations.

- Domain
- Obstacles
- Targets
- Agents
"""
import logging
import multiprocessing
import os
from functools import partial
from multiprocessing import Process, Event

import numpy as np
from loggingtools import log_with
from matplotlib.path import Path

from crowddynamics.io import load_config, save_data
from crowddynamics.core.integrator import euler_integration
from crowddynamics.core.interactions.interactions import \
    agent_agent_block_list_circular, agent_agent_block_list_three_circle, \
    circular_agent_linear_wall, three_circle_agent_linear_wall
from crowddynamics.core.motion.adjusting import force_adjust_agents, \
    torque_adjust_agents
from crowddynamics.core.motion.fluctuation import force_fluctuation, \
    torque_fluctuation
from crowddynamics.core.steering import static_potential
from crowddynamics.core.steering.navigation import to_indices
from crowddynamics.core.structures.agents import is_model, reset_motion
from crowddynamics.core.structures.obstacles import geom_to_linear_obstacles
from crowddynamics.core.vector import angle_nx2
from crowddynamics.simulation.taskgraph import TaskNode

BASE_DIR = os.path.dirname(__file__)
AGENT_CFG_SPEC = os.path.join(BASE_DIR, 'multiagent_spec.cfg')
AGENT_CFG = os.path.join(BASE_DIR, 'multiagent.cfg')
CONFIG = load_config(AGENT_CFG, AGENT_CFG_SPEC)


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

        self.queue = multiprocessing.Queue()
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
        """Tasks"""
        return self.__tasks

    @tasks.setter
    def tasks(self, tasks):
        """Set task graph to the simulation

        Args:
            tasks (TaskNode):
        """
        self.__tasks = tasks

    @log_with(logger)
    def update(self):
        """Execute new iteration cycle of the simulation."""
        self.tasks.evaluate()
        self.iterations += 1

    def __str__(self):
        return self.name


class MultiAgentProcess(Process):
    """MultiAgentProcess

    Class for running MultiAgentSimulation in a new process.
    """
    logger = logging.getLogger(__name__)
    # End of Simulation. Value that is injected into queue when simulation ends.
    EOS = None

    def __init__(self, simulation, maxiter=None):
        """Init MultiAgentProcess

        Args:
            simulation (MultiAgentSimulation):
        """
        super(MultiAgentProcess, self).__init__()
        self.simulation = simulation
        self.exit = Event()
        self.maxiter = maxiter

    @log_with(logger)
    def run(self):
        """Runs simulation process by calling update method repeatedly until
        stop is called. This method is called automatically by Process class
        when start is called."""
        while not self.exit.is_set():
            self.simulation.update()
            if self.maxiter and self.simulation.iterations > self.maxiter:
                self.stop()
        self.simulation.queue.put(self.EOS)

    @log_with(logger)
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
def run_parallel(*simulations, maxiter=None):
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
        process = MultiAgentProcess(simulation, maxiter)
        process.start()
        yield process


class MASTaskNode(TaskNode):
    def __init__(self, simulation):
        """MultiAgentSimulation TaskNode
        
        Args:
            simulation (MultiAgentSimulation): 
        """
        super(MASTaskNode, self).__init__()
        assert isinstance(simulation, MultiAgentSimulation)
        self.simulation = simulation


class Integrator(MASTaskNode):
    r"""Integrator"""

    def __init__(self, simulation):
        super().__init__(simulation)
        self.dt = CONFIG['integrator']['dt_min'], \
                  CONFIG['integrator']['dt_max']
        self.time_tot = np.float64(0.0)
        self.dt_prev = np.float64(np.nan)

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
        if is_model(self.simulation.agents_array, 'three_circle'):
            agent['torque'] += torque_fluctuation(agent['inertia_rot'],
                                                  agent['std_rand_torque'])


class Adjusting(MASTaskNode):
    r"""Adjusting"""

    def update(self):
        agents = self.simulation.agents_array
        force_adjust_agents(agents)
        if is_model(self.simulation.agents_array, 'three_circle'):
            torque_adjust_agents(agents)


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
        self.step = CONFIG['navigation']['step']
        self.direction_map = static_potential(
            self.step, self.simulation.domain, self.simulation.targets,
            self.simulation.obstacles,
            radius=CONFIG['navigation']['radius'],
            value=CONFIG['navigation']['value'])

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
            dir_to_orient = angle_nx2(
                self.simulation.agents_array['target_direction'])
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


class SaveAgentsData(MASTaskNode):
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


class Contains(MASTaskNode):
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
