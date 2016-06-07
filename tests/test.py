import numpy as np

from src.params import Params
from src.core.force import force_social_naive, force_social
from src.struct.agent import agent_struct
from src.struct.constant import Constant
from src.struct.wall import LinearWall, RoundWall


np.set_printoptions(precision=5, threshold=100, edgeitems=3, linewidth=75,
                    suppress=True, nanstr=None, infstr=None,
                    formatter=None)


size = 300
params = Params(50, 50)

constant = Constant()

agent = agent_struct(*params.agent(size))

linear_wall = LinearWall(params.linear_wall(10))
round_wall = RoundWall(params.round_wall(10, 0.1, 0.3))

params.random_position(agent.position, agent.radius, walls=linear_wall)
agent.velocity = params.random_unit_vector(agent.size)


for i in range(agent.size):
    for w in range(linear_wall.size):
        x = agent.position[i]
        v = agent.velocity[i]
        radius = agent.get_radius(i)

        distance, normal = linear_wall.distance_with_normal(w, agent.position[i])
        relative_distance = radius - distance
        relative_position = linear_wall.relative_position(w, x, v)

        f0 = force_social_naive(relative_distance, normal, constant.a, constant.b)
        f1 = force_social(relative_position, v, radius, constant.k, constant.tau_0)

        if np.hypot(f1[0], f1[1]) > 0:
            print("distance:", abs(relative_distance), f0, f1)
