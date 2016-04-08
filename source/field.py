from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def round_wall(p_0, r):
    # TODO: Arc
    p_0 = np.array(p_0)
    return p_0, r


def linear_wall(p_0, p_1):
    p_0 = np.array(p_0)
    p_1 = np.array(p_1)

    rot90 = np.array([[0, -1], [1,  0]])  # 90 degree counterclockwise rotation
    d = p_0 - p_1
    l_w = np.sqrt(np.dot(d, d))  # Length
    t_w = d / l_w  # Tangent (unit)vector
    n_w = np.dot(rot90, t_w)  # Normal (unit)vector
    inv_a = - np.stack((t_w, n_w), axis=1)  # [-n_w, t_w]^(-1) = - [t_w, n_w]
    return p_0, p_1, inv_a, l_w, n_w


def set_walls(round_w, linear_w):
    wall = {
        'round': map(round_wall, round_w),
        'linear': map(linear_wall, linear_w)
    }
    return wall


def populate_agents(amount):
    """
    Populate the positions of the agents in to the field so that they don't
    overlap each others or the walls.
    """
    pass


def set_agents(amount, x_dims, y_dims, mass, radii):
    np.random.seed()

    agent = {
        'mass': np.random.uniform(*mass, size=amount),
        'radius': np.random.uniform(*radii, size=amount),
        'position': None,
        'velocity': None,
        'goal_velocity': None
    }

    # Variables
    orientation = np.random.uniform(0, 2 * np.pi, amount)
    position = np.stack((np.random.uniform(*x_dims, size=amount),
                         np.random.uniform(*y_dims, size=amount)), axis=1)
    velocity = np.stack((np.cos(orientation), np.sin(orientation)), axis=1)

    agent['position'] = position
    agent['velocity'] = velocity
    agent['goal_velocity'] = 1.5 * np.copy(velocity)
    return agent
