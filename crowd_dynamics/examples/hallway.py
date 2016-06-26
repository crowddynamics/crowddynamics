import numpy as np

from crowd_dynamics.area import GoalRectangle
from crowd_dynamics.parameters import Parameters
from crowd_dynamics.structure.agent import Agent
from crowd_dynamics.structure.wall import LinearWall


def initialize(size=100, width=30, height=5):
    name = "hallway"

    parameters = Parameters(width, height)
    linear_params = np.array((
        ((0.0, 0.0), (width, 0.0)),
        ((0.0, height), (width, height)),
    ))

    walls = LinearWall(linear_params)

    # Goal
    height_ = height / 2
    goals = (
        GoalRectangle(center=(0.0, height_), radius=(1.0, height_)),
        GoalRectangle(center=(width, height_), radius=(1.0, height_))
    )

    # Agents
    agent = Agent(*parameters.agent(size))
    first_half = slice(agent.size // 2)
    second_half = slice(agent.size // 2, None)

    parameters.random_position(
        agent.position[first_half], agent.radius[first_half],
        (1.0, width // 2), (0.0, height), walls)
    parameters.random_position(
        agent.position[second_half], agent.radius[second_half],
        (width // 2, width - 1.0), (0.0, height), walls)

    direction1 = np.array((1.0, 0.0))
    direction2 = np.array((-1.0, 0.0))
    agent.target_direction[first_half] += direction1
    agent.target_direction[second_half] += direction2

    agent.angle[first_half] += 0
    agent.angle[second_half] += np.pi

    agent.update_shoulder_positions()

    return agent, walls, goals, name
