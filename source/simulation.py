# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


# Arrays shape of (agent_num, 2) for x and y-components
# position = np.array([])  # Center of agent
# velocity = np.array([])  # Current velocity of agent
# goal_velocity = np.array([])  # Goal velocity of agent

# Agent Parameters
# radius = .2  # Collision radius
# sight = 7  # Neighbor search range
# force_max = 5  # Maximum force/acceleration


def init_simulation(agents_num, size):
    """

    :param agents_num:
    :param size:
    :return:
    """
    tau = 0.5
    radius = 0.2

    shape = (agents_num, 2)  # rows, cols
    angle = np.random.uniform(0, 2 * np.pi, agents_num)

    position = np.random.uniform(0, size, shape)
    velocity = np.stack((np.cos(angle), np.sin(angle)), axis=1)
    goal_velocity = 1.5 * np.copy(velocity)
    # mass = np.random.uniform(60, 90, agents_num)

