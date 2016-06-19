import os
import sys
from collections import namedtuple

import numpy as np

sys.path.append("/home/jaan/Dropbox/Projects/Crowd-Dynamics")
from crowd_dynamics.parameters import Parameters
from crowd_dynamics.struct.agent import Agent
from crowd_dynamics.area import GoalRectangle
from crowd_dynamics.struct.constant import Constant
from crowd_dynamics.struct.wall import LinearWall

# Path to this folder
filepath = os.path.abspath(__file__)
dirpath, name = os.path.split(filepath)
dirpath = os.path.join(os.path.dirname(dirpath), "results")
name, _ = os.path.splitext(name)


def initialize():
    # Field
    dim = namedtuple('dim', ['width', 'height'])
    lim = namedtuple('lim', ['min', 'max'])
    d = dim(50.0, 5.0)
    x = lim(0.0, d.width)
    y = lim(0.0, d.height)

    parameters = Parameters(*d)
    constant = Constant()
    linear_params = np.array((
        ((x.min - 5, y.min), (x.max + 5, y.min)),
        ((x.min - 5, y.max), (x.max + 5, y.max)),
    ))

    linear_wall = LinearWall(linear_params)
    walls = linear_wall

    # Agents
    size = 150
    agent = Agent(*parameters.agent(size))
    agent.three_circles_flag = True

    first_half = slice(agent.size // 2)
    second_half = slice(agent.size // 2, None)

    parameters.random_position(agent.position[first_half], agent.radius,
                               (x.min, x.max // 2), y, walls)
    parameters.random_position(agent.position[second_half], agent.radius,
                               (x.max // 2, x.max), y, walls)

    # Goal
    agent.target_direction[first_half] += np.array((1, 0), dtype=np.float64)
    agent.target_direction[second_half] += np.array((-1, 0), dtype=np.float64)

    goal = GoalRectangle(center=np.array((x.max + 2.5, y.max / 2)),
                         radius=np.array((2.5, d.height / 2)))

    goal2 = GoalRectangle(center=np.array((x.min - 2.5, y.max / 2)),
                          radius=np.array((2.5, d.height / 2)))

    goals = goal, goal2

    return constant, agent, walls, goals, dirpath, name, x, y
