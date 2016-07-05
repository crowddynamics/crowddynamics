import os

import numpy as np
import pandas as pd
from scipy.stats import truncnorm as tn

from crowd_dynamics.environment import Area
from crowd_dynamics.core.vector2d import length_nx2
from crowd_dynamics.functions import filter_none


def load_tables():
    root = os.path.abspath(__file__)
    root, _ = os.path.split(root)
    folder = "tables"

    path1 = os.path.join(root, folder, "body_types.csv")
    path2 = os.path.join(root, folder, "agent_table.csv")

    # TODO: Values converters.
    body_types = pd.read_csv(path1, index_col=[0])
    agent_table = pd.read_csv(path2, index_col=[0])
    return body_types, agent_table


def populate(agent, amount, area: Area, walls=None):
    """
    Monte Carlo method for filling an area with desired amount of circles.

    Loop:
    #) Generate a random element inside desired area.
    #) Check if overlapping with
        #) Agents
        #) Walls
    #) Save value

    :param walls:
    :param amount:
    :param area: Area to be filled.
    :param wall:
    """
    # Fill inactive agents
    inactive = agent.active ^ True
    radius = agent.radius[inactive][:amount]
    position = agent.position[inactive][:amount]
    indices = np.arange(agent.size)[inactive][:amount]

    area_agent = np.sum(np.pi * radius ** 2)
    fill_rate = area_agent / area.size()
    threshold = 0.8
    print("Monte Carlo fill rate: {}".format(fill_rate))
    if fill_rate > threshold:
        print(Warning("High fill rate of {} over threshold {}.\n"
                      "Cannot guarantee of placement of all agents.".format(
            fill_rate, threshold)))

    walls = filter_none(walls)

    i = 0  # Number of agents placed
    iterations = 0  # Number of iterations done
    maxlen = len(position)
    while i < maxlen and iterations < 10 * maxlen:
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

    print("Number of agents placed: {}".format(i))


class Parameters:
    """
    Generates random parameters for simulations and testing.
    """

    def __init__(self, width, height):
        self.x = (0.0, width)
        self.y = (0.0, height)

    @staticmethod
    def truncnorm(loc, abs_scale, size, std=3.0):
        """Scaled symmetrical truncated normal distribution."""
        return np.array(tn.rvs(-std, std, loc=loc, scale=abs_scale / std, size=size))

    @staticmethod
    def random_unit_vector(size):
        """Random unit vector."""
        orientation = np.random.uniform(0, 2 * np.pi, size=size)
        return np.stack((np.cos(orientation), np.sin(orientation)), axis=1)

    def random_2d_coordinates(self, size):
        """Random x and y coordinates inside dims."""
        return np.stack((np.random.uniform(*self.x, size=size),
                         np.random.uniform(*self.y, size=size)), axis=1)

    def random_round_wall(self, size, r_min, r_max):
        """Arguments for constructing round wall."""
        return np.stack((np.random.uniform(*self.x, size=size),
                         np.random.uniform(*self.y, size=size),
                         np.random.uniform(r_min, r_max, size=size)), axis=1)

    def random_linear_wall(self, size):
        """Arguments for constructing linear wall."""
        args = zip(self.random_2d_coordinates(size),
                   self.random_2d_coordinates(size))
        return np.array(tuple(args))

    def agent(self, size, body_type="adult"):
        """Arguments for constructing agent."""
        body_types, agent_table = load_tables()
        body = body_types[body_type]
        values = agent_table["value"]

        mass = self.truncnorm(body["mass"], body["mass_scale"], size)
        radius = self.truncnorm(body["radius"], body["dr"], size)
        r_t = body["k_t"] * radius    # radius_torso
        r_s = body["k_s"] * radius    # radius_shoulder
        r_ts = body["k_ts"] * radius  # distance_torso_shoulder

        # target_velocity = body['v'] * np.ones(size)
        target_velocity = self.truncnorm(body['v'], body['dv'], size)

        # TODO: converters. Eval to values.
        pi = np.pi  # For eval
        inertia_rot = eval(values["inertia_rot"]) * np.ones(size)
        target_angular_velocity = eval(values["target_angular_velocity"]) * np.ones(size)

        return size, mass, radius, r_t, r_s, r_ts, inertia_rot, target_velocity, target_angular_velocity


