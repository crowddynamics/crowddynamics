"""Structure numpy.dtypes"""
from collections import namedtuple

import numpy as np


# TODO: replace
obstacle_type_linear = np.dtype([
    ('p0', np.float64, 2),
    ('p1', np.float64, 2),
])


Neighborhood = namedtuple('Neighborhood',
                          ['neighbor_radius', 'neighborhood_size', 'neighbors'])


def init_neighborhood(agent_size, neighborhood_size, neighbor_radius):
    """Initialise neighborhood

    Args:
        agent_size (int):
        neighborhood_size (int):
        neighbor_radius (float):

    Returns:
        Neighborhood:
    """
    dtype = np.dtype([
        ('agent_indices', np.int64, neighborhood_size),
        ('distances', np.float64, neighborhood_size),
        ('distances_max', np.float64),
    ])
    neighbors = np.zeros(agent_size, dtype=dtype)
    neighborhood = Neighborhood(neighbor_radius, neighborhood_size, neighbors)
    reset_neighborhood(neighborhood)
    return neighborhood


def reset_neighborhood(neighborhood):
    missing = -1
    neighborhood.neighbors['agent_indices'] = missing
    neighborhood.neighbors['distances'] = np.inf
    neighborhood.neighbors['distances_max'] = np.inf
