import logging
from collections import Iterable
from copy import deepcopy
from multiprocessing import Process, Event, Queue

import numpy as np
from scipy.stats import truncnorm as tn

from src.config import Load
from src.core.interactions import agent_agent, agent_wall
from src.core.motion import force_adjust, force_fluctuation, \
    torque_adjust, torque_fluctuation
from src.core.motion import integrator
from src.core.navigation import Navigation, Orientation
from src.core.vector2d import angle_nx2, length_nx2
from src.functions import filter_none
from src.io.hdfstore import HDFStore
from src.multiagent.agent import Agent
from src.multiagent.surface import Area


def random_unit_vector(size):
    """Random unit vector."""
    orientation = np.random.uniform(0, 2 * np.pi, size=size)
    return np.stack((np.cos(orientation), np.sin(orientation)), axis=1)


def truncnorm(loc, abs_scale, size, std=3.0):
    """Scaled symmetrical truncated normal distribution."""
    scale = abs_scale / std
    return tn.rvs(-std, std, loc=loc, scale=scale, size=size)


def agent_motion(indices: (slice, np.ndarray),
                 agent: Agent,
                 target_direction: np.ndarray = None,
                 target_angle: np.ndarray = None,
                 velocity: np.ndarray = None,
                 body_angle: float = None):
    """Set initial parameters for motion.
    :param indices:
    :param agent:
    :param target_direction:
    :param target_angle:
    :param velocity:
    :param body_angle:
    :return:
    """
    logging.info("")
    # FIXME: Nones
    # TODO: 'random' param

    if target_direction is not None:
        agent.target_direction[indices] = target_direction

    if velocity is None and target_direction is not None:
        agent.velocity[indices] = agent.target_direction[indices]
        agent.velocity[indices] *= agent.target_velocity[indices]
    elif velocity is not None:
        agent.velocity[indices] = velocity

    if target_angle is None and target_direction is not None:
        agent.target_angle[indices] = angle_nx2(agent.target_direction[indices])
    elif target_angle is not None:
        agent.target_angle[indices] = target_angle

    if body_angle is None and target_direction is not None:
        agent.angle[indices] = angle_nx2(agent.velocity[indices])
    elif body_angle is not None:
        agent.angle[indices] = body_angle


def agent_positions(agent: Agent,
                    amount: int,
                    area: Area,
                    walls=None,
                    domain: Area = None,
                    target_direction: np.ndarray = None,
                    target_angle: np.ndarray = None,
                    velocity: np.ndarray = None,
                    body_angle: float = None, ):
    """Monte Carlo method for filling an area with desired amount of circles."""
    # Fill inactive agents
    inactive = agent.active ^ True
    radius = agent.radius[inactive][:amount]
    position = agent.position[inactive][:amount]
    indices = np.arange(agent.size)[inactive][:amount]

    walls = filter_none(walls)

    i = 0  # Number of agents placed
    iterations = 0  # Number of iterations done
    maxlen = len(position)
    maxiter = 10 * maxlen
    while i < maxlen and iterations < maxiter:
        iterations += 1
        pos = area.random()

        rad = radius[i]
        radii = radius[:i]

        if domain is not None and not domain.contains(pos):
            continue

        # Test overlapping with other agents
        if i > 0:
            d = length_nx2(pos - position[:i]) - (rad + radii)
            cond = np.all(d > 0)
            if not cond:
                continue

        # Test overlapping with walls
        cond = 1
        for wall in walls:
            for j in range(wall.size):
                d = wall.distance(j, pos) - rad
                cond *= d > 0
        if not cond:
            continue

        position[i] = pos

        index = indices[i]
        agent.position[index] = pos
        agent.active[index] = True
        i += 1

    logging.info("Iterations: {}/{}".format(iterations, maxiter))
    logging.info("Agents placed: {}/{}".format(i, maxlen))

    # FIXME
    agent_motion(indices[:i], agent, target_direction, target_angle,
                 velocity, body_angle)


class QueueDict:
    def __init__(self, producer):
        self.producer = producer
        self.dict = {}

    def set(self, args):
        self.dict.clear()
        for key, attrs in args:
            self.dict[key] = {}
            for attr in attrs:
                self.dict[key][attr] = None

    def fill(self, d):
        for key, attrs in d.items():
            item = getattr(self.producer, key)
            for attr in attrs.keys():
                d[key][attr] = np.copy(getattr(item, attr))

    def get(self):
        d = deepcopy(self.dict)
        self.fill(d)
        return d


class MultiAgentSimulation(Process):
    structures = ("domain", "goals", "exits", "walls", "agent")
    parameters = ("dt_min", "dt_max", "time_tot", "in_goal", "dt_prev")

    def __init__(self, queue: Queue=None):
        # Multiprocessing
        super(MultiAgentSimulation, self).__init__()
        self.queue = queue
        self.exit = Event()

        # Structures
        self.domain = None  # Area
        self.goals = ()     # Area
        self.exits = ()     # Exit
        self.walls = ()     # Obstacle
        self.agent = None   # Agent

        # Angle and direction update algorithms
        self.navigation = None
        self.orientation = None

        # Additional models
        self.game = None

        # Integrator timestep
        self.dt_min = 0.001
        self.dt_max = 0.01

        # State of the simulation
        self.iterations = 0  # Integer
        self.time_tot = 0.0  # Float (types matter for saving to a file)
        self.in_goal = 0     # Integer TODO: Move to area?
        self.dt_prev = 0.1   # Float. Last used time step.

        # Data
        self.load = Load()
        self.hdfstore = None
        self.queue_dict = None

    @property
    def name(self):
        return self.__class__.__name__

    def stop(self):
        logging.info("MultiAgent Exit...")
        self.exit.set()

    def run(self):
        logging.info("MultiAgent Starting")
        while not self.exit.is_set():
            self.update()
        self.queue.put(None)  # Poison pill. Ends simulation
        logging.info("MultiAgent Stopping")

    def configure_domain(self, domain):
        logging.info("In: {}".format(domain))
        if isinstance(domain, Area):
            self.domain = domain
        elif domain is None:
            # Full real domain
            raise NotImplemented("Full real space is not yet supported.")
        else:
            logging.warning("")
            raise ValueError("Domain is wrong type.")
        logging.info("Out: {}".format(domain))

    def configure_goals(self, goals=None):
        logging.info("In: {}".format(goals))
        if goals is None:
            self.goals = ()
        elif isinstance(goals, Iterable):
            self.goals = goals
        else:
            self.goals = (goals,)
        logging.info("Out: {}".format(goals))

    def configure_obstacles(self, obstacles=None):
        logging.info("In: {}".format(obstacles))
        if obstacles is None:
            self.walls = ()
        elif isinstance(obstacles, Iterable):
            self.walls = obstacles
        else:
            self.walls = (obstacles,)
        logging.info("Out: {}".format(obstacles))

    def configure_exits(self, exits=None):
        logging.info("In: {}".format(exits))
        if exits is None:
            self.exits = ()
        elif isinstance(exits, Iterable):
            self.exits = exits
        else:
            self.exits = (exits,)
        logging.info("Out: {}".format(exits))

    def configure_agent(self, size, body):
        logging.info("In: {}, {}".format(size, body))

        # Load tabular values
        body = self.load.csv("body")[body]
        values = self.load.csv("agent")["value"]

        # Eval
        pi = np.pi

        # Arguments for Agent
        mass = truncnorm(body["mass"], body["mass_scale"], size)
        radius = truncnorm(body["radius"], body["dr"], size)
        radius_torso = body["k_t"] * radius
        radius_shoulder = body["k_s"] * radius
        torso_shoulder = body["k_ts"] * radius
        target_velocity = truncnorm(body['v'], body['dv'], size)
        inertia_rot = eval(values["inertia_rot"]) * np.ones(size)
        target_angular_velocity = eval(values["target_angular_velocity"]) * \
                                  np.ones(size)

        # Agent class
        self.agent = Agent(size, mass, radius, radius_torso, radius_shoulder,
                           torso_shoulder, inertia_rot, target_velocity,
                           target_angular_velocity)

    def configure_agent_model(self, model):
        logging.info("In: {}".format(model))
        if model == "circular":
            self.agent.set_circular()
        elif model == "three_circle":
            self.agent.set_three_circle()
        else:
            logging.warning("")
            raise Warning()
        logging.info("Out")

    def configure_agent_positions(self, kwargs):
        logging.info("")

        # TODO: separate, manual positions
        # Initial positions
        if isinstance(kwargs, dict):
            agent_positions(self.agent, walls=self.walls, domain=self.domain,
                            **kwargs)
        elif isinstance(kwargs, Iterable):
            for kwarg in kwargs:
                agent_positions(self.agent, walls=self.walls,
                                domain=self.domain, **kwarg)
        else:
            logging.warning("")
            raise ValueError("")

        self.agent.update_shoulder_positions()
        logging.info("")

    def configure_navigation(self, custom=None):
        """Default navigation algorithm"""
        if custom is None:
            self.navigation = Navigation(self.agent, self.domain, self.walls,
                                         self.exits)
        else:
            self.navigation = custom
        logging.info("")

    def configure_orientation(self, custom=None):
        """Default orientation algorithm"""
        if custom is None:
            self.orientation = Orientation(self.agent)
        else:
            self.orientation = custom
        logging.info("")

    def configure_hdfstore(self):
        if self.hdfstore is None:
            logging.info("")

            # Configure hdfstore file
            self.hdfstore = HDFStore(self.name)

            # Add dataset
            parameters = self.load.yaml('parameters')

            args = self.agent, parameters['agent']
            self.hdfstore.add_dataset(*args)
            self.hdfstore.add_buffers(*args)

            args = self, parameters['simulation']
            self.hdfstore.add_dataset(*args)
            self.hdfstore.add_buffers(*args)

            logging.info("")
        else:
            logging.info("Already configured.")

    def configure_queuing(self, args):
        """

        :param args: Example [("agent", ["position", "active", "position_ls", "position_rs"])]
        :return:
        """
        # FIXME
        if self.queue is not None:
            logging.info("")
            self.queue_dict = QueueDict(self)
            self.queue_dict.set(args)
        else:
            logging.info("Queue is not defined.")

    def update(self):
        logging.debug("")

        # Path finding and rotation planning
        if self.navigation is not None:
            self.navigation.update()

        if self.orientation is not None and self.agent.orientable:
            self.orientation.update()

        # Computing motion (forces and torques) for the system
        self.agent.reset_motion()
        self.agent.reset_neighbor()

        force_adjust(self.agent)
        force_fluctuation(self.agent)
        if self.agent.orientable:
            torque_adjust(self.agent)
            torque_fluctuation(self.agent)
        agent_agent(self.agent)
        for wall in self.walls:
            agent_wall(self.agent, wall)

        # Integration of the system
        self.dt_prev = integrator(self.agent, self.dt_min, self.dt_max)
        self.time_tot += self.dt_prev

        # Game theoretical model
        if self.game is not None:
            self.game.update(self.time_tot, self.dt_prev)

        # Check which agent are inside the domain aka active
        if self.domain is not None:
            self.agent.active &= self.domain.contains(self.agent.position)

        # Check which agent have reached their desired goals
        for goal in self.goals:
            num = -np.sum(self.agent.goal_reached)
            self.agent.goal_reached |= goal.contains(self.agent.position)
            num += np.sum(self.agent.goal_reached)
            self.in_goal += num

        # Raise iteration count
        self.iterations += 1

        # Stores the simulation data into buffers and dumps buffer into file
        if self.hdfstore is not None:
            self.hdfstore.update_buffers()
            if self.iterations % 100 == 0:
                self.hdfstore.dump_buffers()

        data = self.queue_dict.get()
        self.queue.put(data)
