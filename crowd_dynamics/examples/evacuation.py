from functools import partial

import numba
import numpy as np

from crowd_dynamics.core.vector2d import rotate90, normalize, length
from crowd_dynamics.parameters import Parameters
from crowd_dynamics.simulation import Simulation
from crowd_dynamics.structure.agent import Agent
from crowd_dynamics.structure.environment import Goal
from crowd_dynamics.structure.wall import LinearWall


@numba.jit(nopython=True)
def _direction_update(agent, target, mid, r_mid, c_rect, r_rect):
    target_direction = np.zeros(agent.shape)
    for i in range(agent.size):
        x = agent.position[i]
        cond1 = (x - c_rect) <= r_rect
        cond2 = length(mid - x) > r_mid
        if np.all(cond1) and cond2:
            # if inside walls and not inside circle near exit
            target_direction[i] = normalize(target - x)
        else:
            target_direction[i] = np.array((1.0, 0.0))
    return target_direction


def initialize(size=100, width=10, height=10, door_width=1.2, exit_hall_width=1,
               path="", **kwargs):
    name = "evacuation"

    parameters = Parameters(width, height)

    corner = ((0, 0), (0, height), (width, 0), (width, height))
    door = ((width, (height - door_width) / 2),
            (width, (height + door_width) / 2))
    hall = ((width + exit_hall_width, (height - door_width) / 2),
            (width + exit_hall_width, (height + door_width) / 2))

    linear_params = np.array(
        ((corner[0], corner[1]),
         (corner[0], corner[2]),
         (corner[1], corner[3]),
         (corner[2], door[0]),
         (door[1], corner[3]),
         (door[0], hall[0]),
         (door[1], hall[1]),
         ), dtype=np.float64
    )
    walls = LinearWall(linear_params)

    # Goal
    goals = Goal(center=(width + 1.0, height / 2),
                 radius=(1.0, height / 2))

    # Agents
    agent = Agent(*parameters.agent(size))
    parameters.random_position(agent.position, agent.radius,
                               (0.0, width), (0.0, height), walls)

    agent.target_direction += np.array((1.0, 0.0))
    agent.update_shoulder_positions()

    # Navigation algorithm
    door1 = np.array(door[1], dtype=np.float64)
    door0 = np.array(door[0], dtype=np.float64)

    unit = normalize(door1 - door0)
    normal = rotate90(unit)
    mid = (door0 + door1) / 2  # Mid point of the two doors
    r_max = np.max(agent.radius)
    target = mid + r_max * normal
    c_rect = r_rect = np.array((width / 2, height / 2))
    r_mid = door_width / 2

    # print(target, mid, r_mid, c_rect, r_rect)

    direction_update = partial(_direction_update, target=target, mid=mid,
                               r_mid=r_mid, c_rect=c_rect, r_rect=r_rect)

    return Simulation(agent, wall=walls, goals=goals, name=name, dirpath=path,
                      direction_update=direction_update, **kwargs)
