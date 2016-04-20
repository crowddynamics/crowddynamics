from pprint import pprint

import numpy as np

from source.core.force import f_tot_i, acceleration
from source.field import set_field
from source.visualization import plot_field


seed = None

tau_adj = 0.5
k = 1.5
tau_0 = 3.0
sight = 7.0
f_max = 5.0
mu = 1.2e5
kappa = 2.4e5
a = 2e3
b = 0.08

constants = {
    'tau_adj': 0.5,
    'k': 1.5,
    'tau_0': 3.0,
    'sight': 7.0,
    'f_max': 5.0,
    'mu': 1.2e5,
    'kappa': 2.4e5,
    'a': 2e3,
    'b': 0.08
}

field_params = {
    'amount': 100,
    'x_dims': (0, 30),
    'y_dims': (0, 30),
}

agent_params = {
    'mass': (1, 1),
    'radius': (0.2, 0.3),
    'goal_velocity': 1.5
}

wall_params = {
    'round_params':
        (),
    'linear_params': (
        ((0, 30), (4, 30)),
        ((0, 0), (4, 0)),
        ((0, 0), (0, 30))
    )
}

np.random.seed(seed)
agents, walls = set_field(field_params, wall_params, agent_params)

linear_wall = wall_params['linear_params']
round_wall = wall_params['round_params']

x_dims = field_params['x_dims']
y_dims = field_params['y_dims']

m = agents['mass']
r = agents['radius']
x = agents['position']
v = agents['velocity']
v_0 = agents['goal_velocity']
e = agents['goal_direction']


def printing():
    np.set_printoptions()
    print("Params:")
    print(80 * "=")
    pprint(agents)
    pprint(walls)
    print()


def test_force_wall():
    from source.core.force_walls import f_iw_linear_tot

    f = np.zeros_like(x)
    for i in range(len(x)):
        f[i] = f_iw_linear_tot(i, x, v, r, walls['linear_wall'],
                               f_max, sight, mu, kappa, a, b)
    return f


def test_force_agent():
    from source.core.force_agents import f_ij

    f = np.zeros_like(x)
    for i in range(len(x)):
        f[i] = f_ij(i, x, v, r, k, tau_0, sight, f_max, mu, kappa)
    return f


def test_force_adjust():
    pass


def test_force_tot_i():
    a = np.zeros_like(x)
    kwargs = dict(agents, **constants)
    # kwargs = dict(kwargs, **walls)
    kwargs['linear_wall'] = walls['linear_wall']
    a = acceleration(**kwargs)
    return a


if __name__ == '__main__':
    force = None
    # force = test_force_agent()
    # force = test_force_wall()
    force = test_force_tot_i()
    plot_field(x, r, x_dims, y_dims, linear_wall, force, save=True)
