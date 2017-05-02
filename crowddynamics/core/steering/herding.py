"""Herding / Flocking / Leader-Follower effect"""
import numba
import numpy as np
from numba import f8


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
    num = e0_neigh.shape[1]
    _sum = np.zeros(shape=num)
    for i in range(num):
        _sum += e0_neigh[i, :]
    return _sum / num
