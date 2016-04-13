from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


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

system_params = {
    't_delta': 0.01,
}

field_params = {
    'seed': None,
    'amount': 100,
    'x_dims': (0, 4),
    'y_dims': (0, 4),
}

agent_params = {
    'mass': (1, 1),
    'radius': (0.2, 0.2),
    'position': None,
    'velocity': None,
    'goal_velocity': 1.5
}

wall_params = {
    'round_wall':
        [],
    'linear_wall':
        [[[0, 0], [0, 4]],
         [[0, 0], [4, 0]],
         [[0, 4], [4, 4]]]
}

