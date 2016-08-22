import logging
from collections import Iterable, namedtuple
from multiprocessing import Process, Event, Queue

import numpy as np
from scipy.stats import truncnorm as tn

from .io.hdfstore import HDFStore
from .config import Load
from .core.interactions import agent_agent, agent_wall
from .core.motion import force_adjust, force_fluctuation, \
    torque_adjust, torque_fluctuation
from .core.motion import integrator
from .core.navigation import Navigation, Orientation
from .core.vector2d import angle_nx2, length_nx2
from .functions import filter_none, timed
from .structure.agent import Agent
from .structure.area import Area


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
    # FIXME: Nones

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
    """
    Monte Carlo method for filling an area with desired amount of circles.

    Loop:
    #) Generate a random element inside desired area.
    #) Check if overlapping with
        #) Agents
        #) Walls
    #) Save value
    """
    # Fill inactive agents
    inactive = agent.active ^ True
    radius = agent.radius[inactive][:amount]
    position = agent.position[inactive][:amount]
    indices = np.arange(agent.size)[inactive][:amount]

    area_agent = np.sum(np.pi * radius ** 2)
    fill_rate = area_agent / area.size()

    logging.info("Density: {:0.3f}".format(fill_rate))

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

    agent_motion(indices[:i], agent, target_direction, target_angle,
                 velocity, body_angle)

    logging.info("Iterations: {}/{}".format(iterations, maxiter))
    logging.info("Agents placed: {}/{}".format(i, maxlen))


class MultiAgentSimulation(Process):
    def __init__(self, queue: Queue):
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
        self.iterations = 0
        self.time_tot = 0
        self.time_steps = [0]
        self.in_goal = 0  # TODO: In-goal -> Area class
        self.in_goal_time = []

        # Data
        self.hdfstore = None
        self.load = Load()
        self.queuable = {"agent": ("position", "angle", "position_ls",
                                   "position_rs", "front", "active",
                                   "goal_reached")}

    @property
    def name(self):
        return self.__class__.__name__

    def stop(self):
        logging.info("")
        self.exit.set()

    def run(self):
        logging.info("Start")
        while not self.exit.is_set():
            self.update()
        logging.info("End")

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
        logging.info("Out: {}".format(self.agent))

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
            attributes = self.load.yaml('attributes')
            args = self.agent, attributes['agent']
            self.hdfstore.add_dataset(*args)
            self.hdfstore.add_buffers(*args)

            logging.info("")
        else:
            logging.warning("Already configured.")

    def queue_handler(self):
        # FIXME
        data = []
        for struct_name, attrs in self.queuable.items():
            struct = getattr(self, struct_name)
            d = namedtuple(struct_name, attrs)
            for attr in attrs:
                value = np.copy(getattr(struct, attr))
                setattr(d, attr, value)
            data.append(d)
        self.queue.put(data)

    @timed
    def update(self):
        if self.navigation is not None:
            self.navigation.update()

        if self.orientation is not None and self.agent.orientable:
            self.orientation.update()

        # Motion
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

        # Integrate
        dt = integrator(self.agent, self.dt_min, self.dt_max)
        self.time_steps.append(dt)
        self.time_tot += dt

        if self.game is not None:
            self.game.update(self.time_tot, self.time_steps[-1])

        if self.domain is not None:
            self.agent.active &= self.domain.contains(self.agent.position)

        for goal in self.goals:
            num = -np.sum(self.agent.goal_reached)
            self.agent.goal_reached |= goal.contains(self.agent.position)
            num += np.sum(self.agent.goal_reached)
            self.in_goal += num
            for _ in range(num):
                self.in_goal_time.append(self.time_tot)

        self.iterations += 1

        # Put data to the queue
        # self.queue_handler()

        if self.hdfstore is not None:
            self.hdfstore.update_buffers()
            if self.iterations % 100 == 0:
                self.hdfstore.dump_buffers()

        logging.debug("")
