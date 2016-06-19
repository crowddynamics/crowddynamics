# import sys
# sys.path.append("/../Crowd-Dynamics")

import numpy as np

from crowd_dynamics.core.integrator import integrator
from crowd_dynamics.display import timed_execution
from crowd_dynamics.parameters import Parameters
from crowd_dynamics.struct.agent import Agent
from crowd_dynamics.struct.constant import Constant
from crowd_dynamics.struct.result import Result
from crowd_dynamics.struct.wall import LinearWall, RoundWall


np.set_printoptions(precision=5, threshold=100, edgeitems=3, linewidth=75,
                    suppress=True, nanstr=None, infstr=None,
                    formatter=None)


size = 200
params = Parameters(50, 50)

result = Result(size)
constant = Constant()

"""Walls"""
linear_wall = LinearWall(params.linear_wall(10))
round_wall = RoundWall(params.round_wall(10, 0.1, 0.3))
walls = (linear_wall, )

"""Agent"""
agent = Agent(*params.agent(size))
params.random_position(agent.position, agent.radius, walls=linear_wall)
agent.velocity = params.random_unit_vector(agent.size)


def test_integrator():
    advance = timed_execution(integrator, 1.0)
    for i in range(200):
        advance(result, constant, agent, walls)
