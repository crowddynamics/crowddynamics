import numpy as np

from crowd_dynamics.area import GoalRectangle
from crowd_dynamics.parameters import Parameters
from crowd_dynamics.structure.agent import Agent
from crowd_dynamics.structure.wall import LinearWall


def initialize(size=200, width=10, height=10, door_width=2):
    # Path and name for saving simulation data
    name = "evacuation"
    path = "/home/jaan/Dropbox/Projects/Crowd-Dynamics-Simulations/results"

    parameters = Parameters(width, height)

    # Field
    x = (0.0, width)
    y = (0.0, height)

    corner = ((0, 0), (0, height), (width, 0), (width, height))
    door = ((width, (height - door_width) / 2),
            (width, (height + door_width) / 2))
    linear_params = np.array(
        ((corner[0], corner[1]),
         (corner[0], corner[2]),
         (corner[1], corner[3]),
         (corner[2], door[0]),
         (door[1], corner[3]),), dtype=np.float64
    )
    walls = LinearWall(linear_params)

    # Goal
    goals = GoalRectangle(center=(52.5, 25.0), radius=(2.5, 5.0))

    # Agents
    agent = Agent(*parameters.agent(size))
    parameters.random_position(agent.position, agent.radius, x, y, walls)
    agent.target_direction += np.array((1.0, 0.0))
    agent.update_shoulder_positions()

    return agent, walls, goals, path, name
