import os
import sys
from collections import namedtuple

sys.path.append("/home/jaan/Dropbox/Projects/Crowd-Dynamics")
from crowd_dynamics.parameters import Parameters
from crowd_dynamics.structure.agent import Agent
from crowd_dynamics.structure.constant import Constant

# Path to this folder
filepath = os.path.abspath(__file__)
dirpath, name = os.path.split(filepath)
dirpath = os.path.join(os.path.dirname(dirpath), "results")
name, _ = os.path.splitext(name)


def initialize():
    # Field
    dim = namedtuple('dim', ['width', 'height'])
    d = dim(50.0, 50.0)

    lim = namedtuple('lim', ['min', 'max'])
    x = lim(0.0, d.width)
    y = lim(0.0, d.height)

    params = Parameters(*d)
    constant = Constant()
    walls = None

    # Agents
    size = 100
    agent = Agent(*params.agent(size))
    params.random_position(agent.position, agent.radius, x, y, walls)
    agent.velocity += agent.goal_velocity * params.random_unit_vector(size)

    # Goal
    goals = None
    agent.goal_direction += params.random_unit_vector(size)

    return constant, agent, walls, goals
