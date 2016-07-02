# import sys
# sys.path.append("/../Crowd-Dynamics")

import numpy as np

from crowd_dynamics.core.motion import integrator
from crowd_dynamics.functions import timed_execution
from crowd_dynamics.parameters import Parameters
from crowd_dynamics.structure.agent import Agent
from crowd_dynamics.structure.result import Result
from crowd_dynamics.structure.wall import LinearWall, RoundWall


np.set_printoptions(precision=5, threshold=100, edgeitems=3, linewidth=75,
                    suppress=True, nanstr=None, infstr=None, formatter=None)


size = 200
params = Parameters(50, 50)

result = Result()

"""Walls"""
linear_wall = LinearWall(params.random_linear_wall(10))
round_wall = RoundWall(params.random_round_wall(10, 0.1, 0.3))
walls = (linear_wall, )

"""Agent"""
agent = Agent(*params.agent(size))
params.random_position(agent.position, agent.radius, walls=linear_wall)
agent.velocity = params.random_unit_vector(agent.size)


def test_integrator():
    advance = timed_execution(integrator)
    for i in range(200):
        advance(result, agent, walls)
