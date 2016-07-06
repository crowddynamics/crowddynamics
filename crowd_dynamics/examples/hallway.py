import numpy as np

from crowd_dynamics.environment import Rectangle
from crowd_dynamics.parameters import Parameters, populate
from crowd_dynamics.simulation import Simulation
from crowd_dynamics.structure.agent import Agent
from crowd_dynamics.structure.wall import LinearWall


def initialize(size=100, width=30, height=5, path="", **kwargs):
    name = "hallway"

    parameters = Parameters(width, height)
    linear_params = np.array((
        ((0.0, 0.0), (width, 0.0)),
        ((0.0, height), (width, height)),
    ))

    walls = LinearWall(linear_params)

    # Goal
    goals = (
        Rectangle((0, 1), (0, height)),
        Rectangle((width, width+1), (0, height))
    )

    # Agents
    agent = Agent(*parameters.agent(size))
    first_half = slice(agent.size // 2)
    second_half = slice(agent.size // 2, None)

    shape1 = Rectangle((1.0, width // 2), (0.0, height))
    shape2 = Rectangle((width // 2, width - 1.0), (0.0, height))

    populate(agent, agent.size // 2, shape1, walls)
    populate(agent, agent.size // 2, shape2, walls)

    agent.target_direction[first_half] = np.array((1.0, 0.0))
    agent.target_direction[second_half] = np.array((-1.0, 0.0))

    agent.angle[first_half] = 0
    agent.angle[second_half] = np.pi

    agent.update_shoulder_positions()

    return Simulation(agent, wall=walls, goals=goals, name=name, dirpath=path,
                      **kwargs)
