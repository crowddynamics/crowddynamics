from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

try:
    # Find "Source" module to perform import
    module_path = '/home/jaan/Dropbox/Projects/Crowd-Dynamics/'
    sys.path.append(module_path)
except:
    pass

from source.field import set_agents
from source.core.core import update_positions

"""
tau_adj = 0.5  # [s] Characteristic time in which agent adjusts its movement
tau_0 = 3.0  # [s] Max interaction range 2 - 4, aka interaction time horizon
sight = 7.0  # [m] Max distance between agents for interaction to occur
force_max = 5.0  # [N] Forces that are greater will be truncated to max force
mu = 1.2e5  # [kg/s^2]  # Compression counteraction Friction constant
kappa = 2.4e5  # [kg/(m s)]  # Sliding friction constant
a = 2e3  # [N]
b = 0.08  # [m]
"""


def run_simulation():
    constants = {
        'tau_adj': 0.5,
        'tau_0': 3.0,
        'sight': 7.0,
        'force_max': 5.0,
        'mu': 1.2e5,
        'kappa': 2.4e5,
        'a': 2e3,
        'b': 0.08
    }

    params = {
        'mass_range': (70, 90),
        'radii_range': 0.2,
    }

    field = {
        'amount': 40,
        'x_dims': (0, 4),
        'y_dims': (0, 4)
    }
    agents = set_agents(**field)
    simulation = update_positions(agents, constants)
    next(simulation)


run_simulation()
