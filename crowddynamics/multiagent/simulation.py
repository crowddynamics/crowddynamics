"""MultiAgent Simulation"""
import logging
from multiprocessing import Process, Event

import numpy as np
from collections import Iterable
from shapely.geometry import Point, Polygon, GeometryCollection
from shapely.ops import cascaded_union

from crowddynamics.core.interactions import overlapping_three_circle, \
    overlapping_circle_circle
from crowddynamics.core.random.sampling import PolygonSample
from crowddynamics.errors import CrowdDynamicsException, InvalidArgument

from crowddynamics.functions import timed
from crowddynamics.logging import log_with
from crowddynamics.multiagent import Agent
from crowddynamics.multiagent.agent import positions
from crowddynamics.multiagent.parameters import Parameters
from crowddynamics.taskgraph import TaskNode

REGISTERED_SIMULATIONS = []


def agent_polygon(position, radius):
    if isinstance(position, tuple):
        return cascaded_union((
            Point(position[0]).buffer(radius[0]),
            Point(position[1]).buffer(radius[1]),
            Point(position[2]).buffer(radius[2]),
        ))
    else:
        return Point(position).buffer(radius)


class MultiAgentSimulation:
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
        self.name = 'MultiAgentSimulation'
        # Field
        self.domain = Polygon()
        self.obstacles = GeometryCollection()
        self.targets = GeometryCollection()
        # self.agents = dict()
        self.agent = None
        # Currently occupied surface by Agents and Obstacles
        self._occupied = Polygon()
        # Algorithms
        self.queue = None
        self.tasks = None
        self.iterations = int(0)

    def set_name(self, name):
        """Set name for the simulation

        Args:
            name (str): Name string
        """
        self.name = name

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
    def init_agents(self, max_size, model):
        """Initialize agents

        Args:
            max_size (int, optional):
                - ``int``: Maximum number of agents.
                - ``None``: Dynamically increase the size when adding new agents

            model (str):
                Choice from:
                - ``circular``
                - ``three_circle``
        """
        if max_size is None:
            raise NotImplemented
        self.agent = Agent(max_size)
        if model == 'three_circle':
            self.agent.set_three_circle()
        else:
            self.agent.set_circular()

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
        iterations = 0
        sampling = PolygonSample(spawn)
        parameters = Parameters(body_type=body_type)

        while num > 0 and iterations <= iterations_limit * num:
            # Parameters
            position = sampling.draw()
            mass = parameters.mass.default()
            radius = parameters.radius.default()
            ratio_rt = parameters.radius_torso.default()
            ratio_rs = parameters.radius_shoulder.default()
            ratio_ts = parameters.radius_torso_shoulder.default()
            inertia_rot = parameters.moment_of_inertia.default()
            max_velocity = parameters.maximum_velocity.default()
            max_angular_velocity = parameters.maximum_angular_velocity.default()

            # Polygon of the agent
            overlapping_agents = False
            overlapping_obstacles = False
            num_active_agents = np.sum(self.agent.active)
            if num_active_agents > 0:
                # Conditions
                if self.agent.three_circle:
                    r_t = ratio_rt * radius
                    r_s = ratio_rs * radius
                    orientation = 0.0
                    poly = agent_polygon(
                        positions(position, orientation, ratio_rt * radius),
                        (r_t, r_s, r_s)
                    )
                    overlapping_agents = overlapping_three_circle(
                        self.agent, self.agent.indices(),
                        positions(position, orientation, ratio_rt * radius),
                        (r_t, r_s, r_s),
                    )
                else:
                    poly = agent_polygon(position, radius)
                    overlapping_agents = overlapping_circle_circle(
                        self.agent, self.agent.indices(),
                        position,
                        radius
                    )
                overlapping_obstacles = self.obstacles.intersects(poly)

            if not overlapping_agents and not overlapping_obstacles:
                # Add new agent
                index = self.agent.add(
                    position, mass, radius, ratio_rt, ratio_rs, ratio_ts,
                    inertia_rot, max_velocity, max_angular_velocity
                )
                if index >= 0:
                    # Yield index of an agent that was successfully placed.
                    num -= 1
                    # self.agents[index] = poly
                    yield index
                else:
                    break
            iterations += 1

    def remove_agents(self, indices):
        pass

    def set_tasks(self, tasks):
        """Set task graph

        Args:
            tasks (TaskNode):
        """
        self.tasks = tasks

    def set_queue(self, queue):
        """Set queue for the simulation

        Args:
            queue (multiprocessing.Queue):
        """
        self.queue = queue

    def set(self, *args, **kwargs):
        """Method for subclasses to overwrite for setting up simulation."""
        raise NotImplementedError

    @timed()
    def update(self):
        """Execute new iteration cycle of the simulation."""
        self.tasks.evaluate()
        self.iterations += 1

    def register(self):
        """Register simulation in order to make it visible to CLI and GUI."""
        REGISTERED_SIMULATIONS.append(self)

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

    def __init__(self, simulation):
        """Init MultiAgentProcess

        Args:
            simulation (MultiAgentSimulation):
        """
        super(MultiAgentProcess, self).__init__()
        if isinstance(simulation, MultiAgentSimulation):
            raise InvalidArgument("Argument simulation should be instance of"
                                  "MultiAgentSimulation")
        self.simulation = simulation
        self.exit = Event()

    @log_with(logger, entry_msg="starting", exit_msg="Stopped")
    def run(self):
        """Runs simulation process by calling update method repeatedly until
        stop is called. This method is called automatically by Process class
        when start is called."""
        if self.simulation.queue is None:
            raise CrowdDynamicsException("Simulation queue is not set.")

        while not self.exit.is_set():
            self.simulation.update()
        self.simulation.queue.put(self.EOS)

    @log_with(logger, entry_msg="Stopping")
    def stop(self):
        """Sets event to true in order to stop the simulation process."""
        self.exit.set()


def run_simulations(simulations):
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
        process = MultiAgentProcess(simulation)
        process.start()
        yield process
