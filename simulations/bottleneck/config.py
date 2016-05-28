import os
from collections import namedtuple

import numpy as np

from source.struct.agent import agent_struct, random_position
from source.struct.area import GoalRectangle
from source.struct.constant import Constant
from source.struct.wall import LinearWall, RoundWall

# Path to this folder
dirpath = os.path.abspath(__file__)
dirpath = os.path.dirname(dirpath)
name = os.path.basename(dirpath)

goal_point = np.array((53.0, 25.0))


def initialize():
    # Constants
    constant = Constant()

    # Field
    Lim = namedtuple('Lim', ['min', 'max'])
    x = Lim(0, 50)
    y = Lim(0, 50)

    # Walls
    linear_params = np.array(
        (((0, 0), (0, 50)),
         ((0, 0), (50, 0)),
         ((0, 50), (50, 50)),
         ((50, 0), (50, 24)),
         ((50, 26), (50, 50)),), dtype=np.float64
    )

    round_params = np.array(
        ((48, 25, 1),), dtype=np.float64
    )

    # TODO: walls
    linear_wall = LinearWall(linear_params)
    round_wall = None
    walls = (linear_wall, round_wall)

    # Agents
    size = 200
    mass = np.random.normal(loc=70.0, scale=10.0, size=size)
    radius = np.random.normal(loc=0.22, scale=0.01, size=size)
    goal_velocity = 5.0
    agent = agent_struct(size, mass, radius, goal_velocity)
    random_position(agent, x, y, walls)

    # Goal
    goal = GoalRectangle(center=np.array((52.5, 25.0)),
                         radius=np.array((2.5, 5.0)))
    goals = goal

    return constant, agent, walls, goals
