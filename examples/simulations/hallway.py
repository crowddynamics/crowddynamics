from collections import namedtuple

import numpy as np

from crowd_dynamics.area import GoalRectangle
from crowd_dynamics.parameters import Parameters
from crowd_dynamics.structure.agent import Agent
from crowd_dynamics.structure.constant import Constant
from crowd_dynamics.structure.wall import LinearWall


def initialize(size=100, width=30, height=5):
    # Path and name for saving simulation data
    name = "hallway"
    path = "/home/jaan/Dropbox/Projects/Crowd-Dynamics-Simulations/results"

    # Field
    lim = namedtuple('lim', ['min', 'max'])
    x = lim(0.0, width)
    y = lim(0.0, height)

    parameters = Parameters(width, height)
    constant = Constant()
    linear_params = np.array((
        ((x.min - 5, y.min), (x.max + 5, y.min)),
        ((x.min - 5, y.max), (x.max + 5, y.max)),
    ))

    walls = LinearWall(linear_params)

    # Goal
    goals = (
        GoalRectangle(center=(x.max + 2.5, y.max / 2), radius=(2.5, height / 2)),
        GoalRectangle(center=(x.min - 2.5, y.max / 2), radius=(2.5, height / 2))
    )

    # Agents
    agent = Agent(*parameters.agent(size))
    first_half = slice(agent.size // 2)
    second_half = slice(agent.size // 2, None)

    parameters.random_position(agent.position[first_half],
                               agent.radius[first_half],
                               (x.min, x.max // 2),
                               y,
                               walls)
    parameters.random_position(agent.position[second_half],
                               agent.radius[second_half],
                               (x.max // 2, x.max),
                               y,
                               walls)

    direction1 = np.array((1.0, 0.0))
    direction2 = np.array((-1.0, 0.0))
    agent.target_direction[first_half] += direction1
    agent.target_direction[second_half] += direction2
    agent.angle[first_half] += 0
    agent.angle[second_half] += np.pi
    agent.update_shoulder_positions()

    return constant, agent, walls, goals, path, name
