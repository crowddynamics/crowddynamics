import logging as log
from collections import Iterable, namedtuple
from multiprocessing import Process, Event, Queue

import numpy as np
from scipy.stats import truncnorm as tn

from .core.motion import integrator
from .core.navigation import Navigation, Orientation
from .core.vector2d import angle_nx2, length_nx2
from .functions import filter_none, timed
from .io.attributes import Attrs
from .structure.agent import Agent
from .structure.area import Area
from .core.interactions import agent_agent, agent_wall
from .core.motion import force_adjust, force_fluctuation, \
    torque_adjust, torque_fluctuation


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

    log.info("Crowd Density: {}".format(fill_rate))

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

    log.info("Iterations: {}/{}\n"
             "Agents placed: {}/{}".format(iterations, maxiter, i, maxlen))


class MultiAgentSimulation(Process):
    attrs_result = Attrs((
        "dt_min", "dt_max", "iterations", "time_tot", "time_steps",
        "in_goal_time",
    ))

    def __init__(self, queue: Queue):
        # Multiprocessing
        super(MultiAgentSimulation, self).__init__()
        self.queue = queue
        self.exit = Event()

        # Structures
        self.domain = None  # Area
        self.agent = None  # Agent
        self.walls = ()  # Obstacle
        self.goals = ()  # Area
        self.exits = ()  # Exit

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

        # Queuable data
        self.queuable = {"agent": ("position", "angle", "position_ls",
                                   "position_rs", "front", "active",
                                   "goal_reached")}

    @property
    def name(self):
        return self.__class__.__name__

    def stop(self):
        self.exit.set()

    def run(self):
        while not self.exit.is_set():
            self.update()

    def configure_domain(self, domain):
        if isinstance(domain, Area):
            self.domain = domain
            log.info("Domain configured: {}".format(domain))
        elif domain is None:
            # Full real domain
            raise NotImplemented("Full real space is not yet supported.")
        else:
            raise ValueError("Domain is wrong type.")

    def configure_goals(self, goals=None):
        if goals is None:
            self.goals = ()
        elif isinstance(goals, Iterable):
            self.goals = goals
        else:
            self.goals = (goals,)

    def configure_obstacles(self, obstacles=None):
        if obstacles is None:
            self.walls = ()
        elif isinstance(obstacles, Iterable):
            self.walls = obstacles
        else:
            self.walls = (obstacles,)

    def configure_exits(self, exits=None):
        if exits is None:
            self.exits = ()
        elif isinstance(exits, Iterable):
            self.exits = exits
        else:
            self.exits = (exits,)

    def configure_agent(self, size, body):
        # Load tabular values
        from src.data.load import Load
        load = Load()
        body = load.table("body")[body]
        values = load.table("agent")["value"]

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
        log.info("Agent class configured:\n"
                 "Size: {}\n"
                 "Body: {}".format(size, body))

    def configure_agent_model(self, model):
        models = ("circular", "three_circle")

        if model not in models:
            log.warning("Agent model {} not in models {}".format(models, model))
        else:
            if model == "circular":
                self.agent.set_circular()

            if model == "three_circle":
                self.agent.set_three_circle()

            log.info("Agent model configured: {}".format(model))

    def configure_agent_positions(self, kwargs):
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
            raise ValueError("")

        self.agent.update_shoulder_positions()

    def configure_navigation(self, custom=None):
        """Default navigation algorithm"""
        if custom is None:
            self.navigation = Navigation(self.agent, self.domain, self.walls,
                                         self.exits)
        else:
            self.navigation = custom

    def configure_orientation(self, custom=None):
        """Default orientation algorithm"""
        if custom is None:
            self.orientation = Orientation(self.agent)
        else:
            self.orientation = custom

    def queue_handler(self):
        if self.queue.full():
            # TODO: Wait until there is free spaces in the queue
            pass

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
        self.queue_handler()
