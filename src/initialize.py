import logging as log
from collections import Iterable

import numpy as np
from scipy.stats import truncnorm as tn

from src.core.vector2d import length_nx2, angle_nx2
from src.functions import filter_none
from src.structure.agent import Agent
from src.structure.area import Area


def random_unit_vector(size):
    """Random unit vector."""
    orientation = np.random.uniform(0, 2 * np.pi, size=size)
    return np.stack((np.cos(orientation), np.sin(orientation)), axis=1)


def random_2d_coordinates(x, y, size):
    """Random x and y coordinates inside dims."""
    return np.stack((np.random.uniform(*x, size=size),
                     np.random.uniform(*y, size=size)), axis=1)


def random_round_wall(x, y, size, r_min, r_max):
    """Arguments for constructing round wall."""
    return np.stack((np.random.uniform(*x, size=size),
                     np.random.uniform(*y, size=size),
                     np.random.uniform(r_min, r_max, size=size)), axis=1)


def random_linear_wall(x, y, size):
    """Arguments for constructing linear wall."""
    args = zip(random_2d_coordinates(x, y, size),
               random_2d_coordinates(x, y, size))
    return np.array(tuple(args))


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


def initialize_agent(size: int,
                     populate_kwargs_list,
                     body_type="adult",
                     agent_model="circular",
                     walls=None):
    """Arguments for constructing agent."""
    # TODO: converters. Eval to values.

    # Load tabular values
    from src.tables.load import Table
    table = Table()
    body = table.body()[body_type]
    values = table.agent()["value"]

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
    agent = Agent(size,
                  mass,
                  radius,
                  radius_torso,
                  radius_shoulder,
                  torso_shoulder,
                  inertia_rot,
                  target_velocity,
                  target_angular_velocity)

    # Agent model
    agent_models = ("circular", "three_circle")
    default_model = agent_models[0]

    if agent_model not in agent_models:
        log.warning("Model {} not in agent_models {}. "
                    "Using Default: {}".format(
            agent_model, agent_models, default_model))
        agent_model = default_model

    if agent_model == "circular":
        agent.set_circular()

    if agent_model == "three_circle":
        agent.set_three_circle()

    # TODO: separate, manual positions
    # Initial positions
    if isinstance(populate_kwargs_list, dict):
        agent_positions(agent, walls=walls, **populate_kwargs_list)
    elif isinstance(populate_kwargs_list, Iterable):
        for kwargs in populate_kwargs_list:
            agent_positions(agent, walls=walls, **kwargs)
    else:
        raise ValueError("")

    agent.update_shoulder_positions()

    return agent


