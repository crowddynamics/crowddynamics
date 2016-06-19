import os
from collections import namedtuple

import numpy as np

from crowd_dynamics.parameters import Parameters
from crowd_dynamics.struct.agent import Agent
from crowd_dynamics.area import GoalRectangle
from crowd_dynamics.struct.constant import Constant
from crowd_dynamics.struct.wall import LinearWall

# Path to this folder
filepath = os.path.abspath(__file__)
name = os.path.basename(filepath)
dirpath = os.path.join("/home/jaan/Dropbox/Projects/Crowd-Dynamics-Simulations",
                       "results")
name, _ = os.path.splitext(name)


def initialize():
    # Field
    dim = namedtuple('dim', ['width', 'height'])
    lim = namedtuple('lim', ['min', 'max'])
    d = dim(50.0, 50.0)
    x = lim(0.0, d.width)
    y = lim(0.0, d.height)

    constant = Constant()
    linear_params = np.array(
        (((0, 0), (0, 50)),
         ((0, 0), (50, 0)),
         ((0, 50), (50, 50)),
         ((50, 0), (50, 24)),
         ((50, 26), (50, 50)),), dtype=np.float64
    )

    linear_wall = LinearWall(linear_params)
    walls = linear_wall

    # Agents
    size = 100
    parameters = Parameters(*d)
    agent = Agent(*parameters.agent(size))
    parameters.random_position(agent.position, agent.radius, x, y, walls)

    # Goal
    goal = GoalRectangle(center=np.array((52.5, 25.0)),
                         radius=np.array((2.5, 5.0)))
    goals = goal

    return constant, agent, walls, goals
