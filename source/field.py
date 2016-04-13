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
    if p_0 == p_1:
        raise ValueError("{} must not be equal to {}".format(p_0, p_1))
    rot90 = np.array([[0, -1], [1,  0]])  # 90 degree counterclockwise rotation
    p_0 = np.array(p_0)
    p_1 = np.array(p_1)
    d = p_1 - p_0
    l_w = np.sqrt(np.dot(d, d))  # Length of the wall
    t_w = d / l_w                # Tangential (unit)vector
    n_w = np.dot(rot90, t_w)     # Normal (unit)vector
    return p_0, p_1, t_w, n_w, l_w


def set_walls(round_params, linear_params):
    wall = {
        'round': map(round_wall, round_params),
        'linear': map(linear_wall, linear_params)
    }
    return wall


def populate_agents(amount, walls):
    """
    Populate the positions of the agents in to the field so that they don't
    overlap each others or the walls.
    """
    pass


def set_agents(amount, x_dims, y_dims, mass, radius):
    agent = {
        'mass': np.random.uniform(*mass, size=amount),
        'radius': np.random.uniform(*radius, size=amount),
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


def set_field():
    """
    Set Walls and agents.
    """
    np.random.seed()
    pass
