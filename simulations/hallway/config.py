import os
from collections import namedtuple

import numpy as np

from source.struct.agent import agent_struct, random_position
from source.struct.area import GoalRectangle
from source.struct.constant import Constant
from source.struct.wall import LinearWall

# Path to this folder
dirpath = os.path.abspath(__file__)
dirpath = os.path.dirname(dirpath)
name = os.path.basename(dirpath)


# Field
dim = namedtuple('dim', ['width', 'height'])
d = dim(50.0, 5.0)

lim = namedtuple('lim', ['min', 'max'])
x = lim(0.0, d.width)
y = lim(0.0, d.height)


def initialize():
    # Constants
    constant = Constant()

    # Walls
    linear_params = np.array(
        (((x.min - 5, y.min), (x.max + 5, y.min)),
         ((x.min - 5, y.max), (x.max + 5, y.max)),), dtype=np.float64
    )

    linear_wall = LinearWall(linear_params)
    round_wall = None
    walls = (linear_wall, round_wall)

    # Agents
    size = 200
    mass = np.random.normal(loc=70.0, scale=10.0, size=size)
    radius = np.random.normal(loc=0.22, scale=0.01, size=size)
    goal_velocity = 5.0
    agent = agent_struct(size, mass, radius, goal_velocity)

    first_half = slice(agent.size // 2)
    second_half = slice(agent.size // 2, None)

    random_position(agent.position[first_half], agent.radius, (x.min, x.max//2),
                    y, walls)
    random_position(agent.position[second_half], agent.radius, (x.max//2, x.max),
                    y, walls)

    # Goal
    agent.goal_direction[first_half] += np.array((1, 0), dtype=np.float64)
    agent.goal_direction[second_half] += np.array((-1, 0), dtype=np.float64)

    goal = GoalRectangle(center=np.array((x.max + 2.5, y.max / 2)),
                         radius=np.array((2.5, d.height / 2)))

    goal2 = GoalRectangle(center=np.array((x.min - 2.5, y.max / 2)),
                          radius=np.array((2.5, d.height / 2)))

    goals = goal, goal2

    return constant, agent, walls, goals

