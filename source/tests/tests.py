import numpy as np

from source.field.field import set_field
from source.parameters import *
from source.visualization import plot_field


seed = None
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

globals().update(constant)


def test_force_wall():
    from source.core.force_walls import f_iw_tot

    f = np.zeros_like(x)
    for i in range(len(x)):
        f[i] = f_iw_tot(i, x, v, r, walls['linear_wall'],
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
    from source.core.system import f_tot
    a = np.zeros_like(x)
    kwargs = dict(agents, **constant)
    kwargs['linear_wall'] = walls['linear_wall']
    a = f_tot(**kwargs)
    return a


if __name__ == '__main__':
    force = None
    # force = test_force_agent()
    # force = test_force_wall()
    # force = test_force_tot_i()
    plot_field(x, r, x_dims, y_dims, linear_wall, force, save=False)
