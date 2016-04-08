from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

from source import visualization

try:
    # Find "Source" module to perform import
    module_path = '/home/jaan/Dropbox/Projects/Crowd-Dynamics/'
    sys.path.append(module_path)
except:
    pass

from source.field import set_agents, set_walls
from source.core.core import update_positions


def init_simulation():
    constants = {
        'tau_adj': 0.5,
        'tau_0': 3.0,
        'sight': 7.0,
        'f_max': 5.0,
        'mu': 1.2e5,
        'kappa': 2.4e5,
        'a': 2e3,
        'b': 0.08
    }

    wall = {
        'round_w': [],
        'linear_w': [([0, 0], [0, 4]),
                     ([0, 0], [4, 0]),
                     ([0, 4], [4, 4])]
    }

    field = {
        'amount': 100,
        'x_dims': (0, 4),
        'y_dims': (0, 4),
        'mass': (1, 1),
        'radii': (0.2, 0.2),
    }

    walls = set_walls(**wall)
    agents = set_agents(**field)
    simulation = update_positions(agents, constants)
    return simulation, agents, field


if __name__ == '__main__':
    simulation_gen = init_simulation()
    visualization.func_plot(*simulation_gen)
