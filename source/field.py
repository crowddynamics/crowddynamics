from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def populate_agents(amount):
    """
    Populate the positions of the agents in to the field so that they don't
    overlap each others or the walls.
    """
    pass


def set_walls():
    wall = {'circles': None,
            'lines': None}
    return wall


def set_agents(amount, x_dims, y_dims):
    agent = {'mass': np.ones(amount),
             'radius': 0.2 * np.ones(amount),
             'position': None,
             'velocity': None,
             'goal_velocity': None}
    # Variables
    orientation = np.random.uniform(0, 2 * np.pi, amount)
    position = np.stack((np.random.uniform(*x_dims, size=amount),
                         np.random.uniform(*y_dims, size=amount)), axis=1)
    velocity = np.stack((np.cos(orientation), np.sin(orientation)), axis=1)

    agent['position'] = position
    agent['velocity'] = velocity
    agent['goal_velocity'] = 1.5 * np.copy(velocity)
    return agent
