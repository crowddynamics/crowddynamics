from pprint import pprint

import numpy as np

from source.field import set_field
from source.core.force_walls import f_iw_linear_tot


seed = None

tau_adj = 0.5
tau_0 = 3.0
sight = 7.0
f_max = 5.0
mu = 1.2e5,
kappa = 2.4e5
a = 2e3
b = 0.08

field_params = {
    'amount': 100,
    'x_dims': (0, 10),
    'y_dims': (0, 10),
}

agent_params = {
    'mass': 1,
    'radius': (0.2, 0.3),
    # 'position': None,
    # 'velocity': None,
    'goal_velocity': 1.5
}

wall_params = {
    'round_params':
        (),
    'linear_params': (
        ((0, 0), (0, 10)),
        ((0, 0), (10, 0)),
        ((0, 10), (10, 10))
    )
}

np.random.seed(seed)
agents, walls = set_field(field_params, wall_params, agent_params)

linear_wall = walls['linear_wall']
round_wall = walls['round_wall']

m = agents['mass']
r = agents['radius']
x = agents['position']
v = agents['velocity']
v_0 = agents['goal_velocity']
e_i = agents['goal_direction']


np.set_printoptions()
# print("Params:")
# pprint(agents)
# pprint(walls)
# print()


def test_force_wall():
    for i in range(len(x)):
        out = f_iw_linear_tot(i, x, v, r, linear_wall,
                          f_max, sight, mu, kappa, a, b)
        print(out)


def test_force_agent():
    pass


def test_force_adjust():
    pass


if __name__ == '__main__':
    test_force_wall()