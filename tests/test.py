import sys
from timeit import default_timer as timer

import numpy as np
sys.path.append("/home/jaan/Dropbox/Projects/Crowd-Dynamics")
from crowd_dynamics.core.integrator import explicit_euler_method
from crowd_dynamics.display import format_time
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


def timed_execution(gen):
    start = timer()
    ret = next(gen)
    t_diff = timer() - start
    print(format_time(t_diff))
    return ret


def test_integrator():
    integrator = explicit_euler_method(result, constant, agent, walls)
    i = 0
    while i < 100:
        timed_execution(integrator)
        i += 1


test_integrator()
