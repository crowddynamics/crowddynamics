"""Herding / Flocking / Leader-Follower effect"""
import numba
import numpy as np
from numba import f8

from crowddynamics.core.interactions.block_list import block_list


@numba.jit(f8[:](f8[:, :]), nopython=True, nogil=True, cache=True)
def herding(e0_neigh):
    r"""Herding effect. Computed from the average directions of neighbouring 
    agents.

    .. math::
       \mathbf{\hat{e}_{herding}} = 
       \frac{\sum_{j \in Neigh} \mathbf{\hat{e}}_j}{N_{neigh}}

    Args:
        e0 (numpy.ndarray):
            Agents desired direction
            
        e0_neigh (numpy.ndarray):
            Directions of the neighbouring agents

    Returns:
        numpy.ndarray: 
            New direction vector :math:`\mathbf{\hat{e}_{herding}}`
    """
    n_neigh = e0_neigh.shape[1]
    sum_e0_neigh = np.zeros(shape=n_neigh)
    for i in range(n_neigh):
        sum_e0_neigh += e0_neigh[i, :]
    return sum_e0_neigh / n_neigh


def full(agents):
    sight_herding = 3.0
    index_list, count, offset, shape = block_list(agents['position'],
                                                  sight_herding)
