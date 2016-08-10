import logging as log
import os
from collections import Iterable

import numpy as np
from scipy.stats import truncnorm as tn

from .core.vector2d import angle_nx2, length_nx2
from .functions import filter_none
from .core.egress import egress_game_attrs
from .core.motion import motion, integrator
from .core.navigation import navigator
from .functions import timed_execution
from .io.attributes import Intervals, Attrs, Attr
from .io.hdfstore import HDFStore
from .structure.agent import agent_attr_names, Agent
from .structure.area import Area
from .structure.obstacle import wall_attr_names


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


class MultiAgentSimulation:
    def __init__(self):
        self.result_attr_names = (
            "dt_min",
            "dt_max",
            "iterations",
            "time",
            "time_steps",
            "in_goal_time",
        )
        self.attrs_result = Attrs(self.result_attr_names)

        # Integrator timestep
        self.dt_min = 0.001
        self.dt_max = 0.01

        # Angle and direction update algorithms
        self.angle_update = None
        self.direction_update = None

        # State of the simulation
        self.iterations = 0
        self.time = 0
        self.time_steps = [0]
        # TODO: In-Goal to area class
        self.in_goal = 0
        self.in_goal_time = []

        # Structures
        self.domain = None  # None is whole 2-dimensional real domain
        self.agent = None
        self.walls = ()
        self.goals = ()

        # Extras
        self.egress_model = None

        # Saving data to HDF5 file for analysis and resuming a simulation.
        self.hdfstore = None

    @property
    def name(self):
        return self.__class__.__name__

    def configure_saving(self, dirpath):
        log.info("Simulation: Configuring save")

        if dirpath is None or len(dirpath) == 0:
            filepath = self.name
        else:
            os.makedirs(dirpath, exist_ok=True)
            filepath = os.path.join(dirpath, self.name)

        self.hdfstore = HDFStore(filepath)

        attrs_agent = Attrs(agent_attr_names, Intervals(1.0))
        attrs_wall = Attrs(wall_attr_names)

        # TODO: Gui selection for saveable attributes
        attrs = ("position", "velocity", "force", "angle", "angular_velocity",
                 "torque")
        for attr in attrs:
            attrs_agent[attr] = Attr(attr, True, True)

        self.hdfstore.save(self.agent, attrs_agent)
        for w in self.walls:
            self.hdfstore.save(w, attrs_wall)

        if self.egress_model is not None:
            attrs_egress = Attrs(egress_game_attrs, Intervals(1.0))
            attrs_egress["strategy"] = Attr("strategy", True, True)
            attrs_egress["time_evac"] = Attr("time_evac", True, True)
            self.hdfstore.save(self.egress_model, attrs_egress)

    def configure_domain(self, domain):
        if isinstance(domain, Area) or domain is None:
            self.domain = domain
            log.info("Domain configured: {}".format(domain))

    def configure_goals(self, goals):
        if isinstance(goals, Iterable):
            self.walls = goals
        else:
            self.walls = (goals,)

    def configure_agent(self, size, body):
        # Load tabular values
        from src.tables.load import Table
        table = Table()
        body = table.load("body")[body]
        values = table.load("agent")["value"]

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

    def configure_obstacles(self, obstacles):
        if isinstance(obstacles, Iterable):
            self.walls = obstacles
        else:
            self.walls = (obstacles,)

    def configure_agent_positions(self, kwargs):
        # TODO: separate, manual positions
        # Initial positions
        if isinstance(kwargs, dict):
            agent_positions(self.agent, walls=self.walls, **kwargs)
        elif isinstance(kwargs, Iterable):
            for kwarg in kwargs:
                agent_positions(self.agent, walls=self.walls, **kwarg)
        else:
            raise ValueError("")

        self.agent.update_shoulder_positions()

    def load(self):
        pass

    def save(self):
        if self.hdfstore is not None:
            self.hdfstore.save(self, self.attrs_result, overwrite=True)
            self.hdfstore.record(brute=True)

    def exit(self):
        log.info("Simulation exit")
        self.save()

    @timed_execution
    def advance(self):
        # TODO: Initial -> Final
        navigator(self.agent, self.angle_update, self.direction_update)
        motion(self.agent, self.walls)
        dt = integrator(self.agent, self.dt_min, self.dt_max)

        self.iterations += 1
        self.time += dt
        self.time_steps.append(dt)

        if self.egress_model is not None:
            self.egress_model.update(self.time, dt)

        self.agent.reset_neighbor()

        if self.domain is not None:
            self.agent.active &= self.domain.contains(self.agent.position)

        # Goals
        for goal in self.goals:
            num = -np.sum(self.agent.goal_reached)
            self.agent.goal_reached |= goal.contains(self.agent.position)
            num += np.sum(self.agent.goal_reached)
            self.in_goal += num
            self.in_goal_time.extend(num * (self.time,))

        # Save
        if self.hdfstore is not None:
            self.hdfstore.record()

        return True

    def run(self, max_iter=np.inf, max_time=np.inf):
        """

        :param max_iter: Iteration limit
        :param max_time: Time (simulation not real time) limit
        :return: None
        """
        while self.advance() and \
                        self.iterations < max_iter and \
                        self.time < max_time:
            pass
        self.exit()


