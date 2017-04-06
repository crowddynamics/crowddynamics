"""Tools for creating multiagent simulations."""
import logging
import multiprocessing
from collections import Iterable
from multiprocessing import Process, Event

import numpy as np
from loggingtools import log_with
from shapely.geometry import Point, Polygon, GeometryCollection
from shapely.ops import cascaded_union

from crowddynamics.core.interactions.distance import overlapping_three_circle, \
    overlapping_circle_circle
from crowddynamics.core.random.sampling import PolygonSample
from crowddynamics.core.structures.agents import AgentManager
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


def agent_polygon(position, radius):
    if isinstance(position, tuple):
        return cascaded_union((
            Point(position[0]).buffer(radius[0]),
            Point(position[1]).buffer(radius[1]),
            Point(position[2]).buffer(radius[2]),
        ))
    else:
        return Point(position).buffer(radius)


class MultiAgentSimulation(object):
    r"""MultiAgent simulation setup

    1) Set the Field

       - Domain
       - Obstacles
       - Targets (aka exits)

    2) Initialise Agents

       - Set maximum number of agents. This is the limit of the size of array
         inside ``Agent`` class.
       - Select Agent model.

    3) Place Agents into any surface that is contained by the domain.

       - Body type
       - Number of agents that is placed into the surface

    4) Run simulation

    """
    logger = logging.getLogger(__name__)

    def __init__(self):
        # Field
        self.domain = Polygon()
        self.obstacles = GeometryCollection()
        self.targets = GeometryCollection()
        self.agent = None

        # Currently occupied surface by Agents and Obstacles
        self._occupied = Polygon()

        # Algorithms
        self.queue = multiprocessing.Queue()
        self.tasks = None
        self.iterations = 0

    @property
    def name(self):
        """Name of the simulation"""
        return self.__class__.__name__

    @log_with(logger)
    def init_domain(self, domain):
        """Initialize domain

        Args:
            domain (Polygon, optional):
                - ``Polygon``: Subset of real domain
                  :math:`\Omega \subset \mathbb{R}^{2}`.
                - ``None``: Real domain :math:`\Omega = \mathbb{R}^{2}`.
        """
        self.domain = domain

    @log_with(logger)
    def add_obstacle(self, geom):
        """Add new ``obstacle`` to the Field

        Args:
            geom (BaseGeometry):
        """
        self.obstacles |= geom
        self._occupied |= geom

    @log_with(logger)
    def remove_obstacle(self, geom):
        """Remove obstacle"""
        self.obstacles -= geom
        self._occupied -= geom

    @log_with(logger)
    def add_target(self, geom):
        """Add new ``target`` to the Field

        Args:
            geom (BaseGeometry):
        """
        self.targets |= geom

    @log_with(logger)
    def remove_target(self, geom):
        """Remove target"""
        self.targets -= geom

    @log_with(logger)
    def add_agents(self, num, spawn, body_type, iterations_limit=100):
        r"""Add multiple agents at once.

        1) Sample new position from ``PolygonSample``
        2) Check if agent in new position is overlapping with existing ones
        3) Add new agent if there is no overlapping

        Args:
            num (int, optional):
                - Number of agents to be placed into the ``surface``. If given
                  amount of agents does not fit into the ``surface`` only the
                  amount that fits will be placed.
                - ``None``: Places maximum size of agents

            spawn (Polygon, optional):
                - ``Polygon``: Custom polygon that is contained inside the
                  domain
                - ``None``: Domain

            body_type (str):
                Choice from ``Parameter.body_types``:
                - 'adult'
                - 'male'
                - 'female'
                - 'child'
                - 'eldery'

            iterations_limit (int):
                Limits iterations to ``max_iter = iterations_limit * num``.

        Yields:
            int: Index of agent that was placed.

        """
        # Draw random uniformly distributed points from the set on points
        # that belong to the surface. These are used as possible new position
        # for an agents (if it does not overlap other agents).
        sampling = PolygonSample(np.asarray(spawn.exterior))
        position = sampling.draw()
        pass

    @log_with(logger)
    def remove_agents(self, indices):
        pass

    @log_with(logger)
    def set_tasks(self, tasks):
        """Set task graph

        Args:
            tasks (TaskNode):
        """
        self.tasks = tasks

    def set(self, *args, **kwargs):
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
def register(simulation):
    """Register simulation in order to make it visible to CLI and GUI."""
    if isinstance(simulation, MultiAgentSimulation):
        raise InvalidArgument("Argument simulation should be instance of"
                              "MultiAgentSimulation")
    name = simulation.__name__
    if name in REGISTERED_SIMULATIONS:
        raise CrowdDynamicsException('Simulation named: "{name}" already '
                                     'exists in registered simulations.'
                                     'please rename the simulation.')
    REGISTERED_SIMULATIONS[name] = simulation


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
