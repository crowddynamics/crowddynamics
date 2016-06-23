from collections import namedtuple

import numpy as np

from crowd_dynamics.area import GoalRectangle
from crowd_dynamics.parameters import Parameters
from crowd_dynamics.structure.agent import Agent
from crowd_dynamics.structure.wall import LinearWall


def initialize(size=100, width=30, height=5):
    name = "outdoor"

    lim = namedtuple('lim', ['min', 'max'])
    x = lim(0.0, width)
    y = lim(0.0, height)

    params = Parameters(width, height)
    walls = None

    # Agents
    agent = Agent(*params.agent(size))
    params.random_position(agent.position, agent.radius, x, y, walls)
    agent.velocity += agent.goal_velocity * params.random_unit_vector(size)

    # Goal
    goals = None
    agent.goal_direction += params.random_unit_vector(size)

    return agent, walls, goals, name
