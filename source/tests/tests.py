import os
from itertools import zip_longest, repeat, islice
from pprint import pprint

import numpy as np
import collections
from collections import Iterable

from source.field import set_field


def consume(iterator, n=None):
    """Advance the iterator n-steps ahead. If n is none, consume entirely."""
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


seed = 1111

tau_adj = 0.5
tau_0 = 3.0
sight = 7.0
f_max = 5.0
mu = 1.2e5,
kappa = 2.4e5
a = 2e3
b = 0.08

field_params = {
    'amount': 200,
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
        ((0, 10), (4, 10)),
        ((0, 0), (4, 0)),
        ((0, 0), (0, 10))
    )
}


def printing():
    np.set_printoptions()
    print("Params:")
    print(80 * "=")
    pprint(agents)
    pprint(walls)
    print()


def plot_field(x, r, linear_wall, force=None):
    from matplotlib import pyplot as plt
    import seaborn

    def agent_patch(x_, r_):
        return plt.Circle(x_, r_, alpha=0.5)

    def wall_patch(p):
        xdata, ydata = tuple(zip(p[0], p[1]))
        return plt.Line2D(xdata, ydata)

    def force_patch(x_, f):
        return plt.Arrow(x_[0], x_[1], f[0], f[1], width=0.1, alpha=0.7)

    seaborn.set()
    fig, ax = plt.subplots()
    ax.set(xlim=field_params['x_dims'],
           ylim=field_params['y_dims'])

    def add_patch(patch):
        if isinstance(patch, Iterable):
            consume(map(ax.add_artist, patch))
        else:
            ax.add_artist(patch)

    if not isinstance(r, Iterable):
        # If all agents have same radius
        r = repeat(r, times=len(x))

    agents = map(agent_patch, x, r)
    walls = map(wall_patch, linear_wall)
    consume(map(add_patch, agents))
    consume(map(add_patch, walls))
    if force is not None:
        mask = np.hypot(force[:, 0], force[:, 1]) > 0.01
        forces = map(force_patch, x[mask], force[mask])
        consume(map(add_patch, forces))

    # Save figure to figures/field.pdf
    folder = 'figures'
    fname = 'field'
    if not os.path.exists(folder):
        os.mkdir(folder)
    fname = os.path.join(folder, fname)
    plt.savefig(fname + '.pdf')


def test_force_wall(x, v, r, linear_wall,
                    f_max, sight, mu, kappa, a, b):
    from source.core.force_walls import f_iw_linear_tot

    f = np.zeros_like(x)
    for i in range(len(x)):
        f[i] = f_iw_linear_tot(i, x, v, r, linear_wall,
                               f_max, sight, mu, kappa, a, b)
    return f


def test_force_agent():
    from source.core.force_agents import f_ij
    pass


def test_force_adjust():
    from source.core.force import f_adjust_i
    pass


if __name__ == '__main__':
    np.random.seed(seed)
    agents, walls = set_field(field_params, wall_params, agent_params)

    linear_wall = wall_params['linear_params']
    round_wall = wall_params['round_params']

    m = agents['mass']
    r = agents['radius']
    x = agents['position']
    v = agents['velocity']
    v_0 = agents['goal_velocity']
    e = agents['goal_direction']

    force = test_force_wall(x, v, r, walls['linear_wall'],
                            f_max, sight, mu, kappa, a, b)
    plot_field(x, r, linear_wall, force)
