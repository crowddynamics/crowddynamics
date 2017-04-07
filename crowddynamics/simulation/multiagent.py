"""Tools for creating multiagent simulations.

- Domain
- Obstacles
- Targets
- Agents
"""
import logging
import multiprocessing
from collections import Iterable
from multiprocessing import Process, Event

from loggingtools import log_with

from crowddynamics.core.structures.obstacles import geom_to_linear_obstacles
from crowddynamics.exceptions import CrowdDynamicsException, InvalidArgument
from crowddynamics.simulation.taskgraph import TaskNode

__all__ = """
REGISTERED_SIMULATIONS
MultiAgentSimulation
MultiAgentProcess
register
run_simulations_parallel
run_simulations_sequentially
""".split()

REGISTERED_SIMULATIONS = dict()


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

    def setup(self, *args, **kwargs):
        """Method for subclasses to overwrite for setting up simulation."""
        raise NotImplementedError

    @log_with(logger)
    def update(self):
        """Execute new iteration cycle of the simulation."""
        self.tasks.evaluate()
        self.iterations += 1

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def register(self):
        """Register the simulation so it can be found for example by the
        commandline client (CLI)."""
        if self.name in REGISTERED_SIMULATIONS:
            self.logger.warning('Simulation named: "{name}" already '
                                'exists in registered simulations.'.format(
                name=self.name
            ))
        REGISTERED_SIMULATIONS[self.name] = self


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
def run_simulations_parallel(simulations, maxiter=None):
    """Run multiagent simulations as a new process

    Wraps MultiAgentSimulations in MultiAgentProcess class and starts them.
    Started process is yielded and can be stopped with stop method. When
    stop is called process will put ``MultiAgentProcess.EOS`` into the queue as
    sign of final value.

    Args:
        simulations (Iterable[MultiAgentSimulation]):
            Iterable of multiagent simulations to be run in a new process.

    Yields:
        MultiAgentProcess: Started simulation process
    """
    if not isinstance(simulations, Iterable):
        simulations = (simulations,)

    for simulation in simulations:
        process = MultiAgentProcess(simulation, maxiter)
        process.start()
        yield process


@log_with()
def run_simulations_sequentially(simulations, maxiter=None):
    """Run new simulation sequentially

    Args:
        simulation (Iterable[MultiAgentSimulation]):
    """
    if not isinstance(simulations, Iterable):
        simulations = (simulations,)

    for simulation in simulations:
        iterations = 0
        conds = True
        while conds:
            simulation.update()
            iterations += 1
            if maxiter:
                conds = iterations <= maxiter
