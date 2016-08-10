import logging as log
from collections import Iterable

import numpy as np
from scipy.stats import truncnorm as tn

from src.simulation import agent_positions
from src.structure.agent import Agent


def random_2d_coordinates(x, y, size):
    """Random x and y coordinates inside dims."""
    return np.stack((np.random.uniform(*x, size=size),
                     np.random.uniform(*y, size=size)), axis=1)


def truncnorm(loc, abs_scale, size, std=3.0):
    """Scaled symmetrical truncated normal distribution."""
    scale = abs_scale / std
    return tn.rvs(-std, std, loc=loc, scale=scale, size=size)


def initialize_agent(size: int,
                     populate_kwargs_list,
                     body_type="adult",
                     agent_model="circular",
                     walls=None):
    """Arguments for constructing agent."""

    # Load tabular values
    from src.tables.load import Table
    table = Table()
    body = table.load("body")[body_type]
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
    agent = Agent(size, mass, radius, radius_torso, radius_shoulder,
                  torso_shoulder, inertia_rot, target_velocity,
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


