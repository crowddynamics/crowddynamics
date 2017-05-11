"""Herding / Flocking / Leader-Follower effect"""
import numba
import numpy as np
from numba import f8
from numba import i8

from crowddynamics.core.interactions.block_list import block_list
from crowddynamics.core.interactions.block_list import get_block
from crowddynamics.core.vector2D import length, normalize


@numba.jit([f8[:, :](f8[:, :], f8[:, :], f8, i8[:], i8[:], i8[:], i8[:])],
           nopython=True, nogil=True, cache=True)
def herding_interaction(position, direction, sight_herding, index_list, count,
                        offset, shape):
    r"""Herding effect. Computed from the average directions of neighbouring 
    agents.

    .. math::
       \mathbf{\hat{e}_{herding}} = 
       \frac{\sum_{j \in Neigh} \mathbf{\hat{e}}_j}{N_{neigh}}

    Args:
        position (numpy.ndarray):
            Positions of herding agents
        direction (numpy.ndarray):
            Directions (unit vectors) of herding agents
        sight_herding: 
        index_list: 
        count: 
        offset: 
        shape: 

    Returns:
        numpy.ndarray: 
            New direction vector :math:`\mathbf{\hat{e}_{herding}}`
    """
    # Neighbouring blocks
    nb = np.array(((1, 0), (1, 1), (0, 1), (1, -1)), dtype=np.int64)
    n, m = shape

    sum_e0_neigh = np.zeros_like(direction)
    # n_neigh = np.zeros((position.shape[0], 1), dtype=np.int64)

    for i in range(n):
        for j in range(m):
            # Herding between agents inside the block
            ilist = get_block((i, j), index_list, count, offset, shape)
            for l, i_agent in enumerate(ilist[:-1]):
                for j_agent in ilist[l + 1:]:
                    sum_e0_neigh[i_agent] += direction[j_agent]
                    sum_e0_neigh[j_agent] += direction[i_agent]
                    # n_neigh[i_agent] += 1
                    # n_neigh[j_agent] += 1

            # Herding between agent inside the block and neighbouring agents
            for k in range(len(nb)):
                i2, j2 = nb[k]
                if 0 <= (i + i2) < n and 0 <= (j + j2) < m:
                    ilist2 = get_block((i + i2, j + j2), index_list, count,
                                       offset, shape)
                    for i_agent in ilist:
                        for j_agent in ilist2:
                            if length(position[i_agent] - position[j_agent]) \
                                    <= sight_herding:
                                sum_e0_neigh[i_agent] += direction[j_agent]
                                sum_e0_neigh[j_agent] += direction[i_agent]
                                # n_neigh[i_agent] += 1
                                # n_neigh[j_agent] += 1

    # Normalize
    for i in range(len(sum_e0_neigh)):
        sum_e0_neigh[i] = normalize(sum_e0_neigh[i])

    return sum_e0_neigh


def herding_block_list(position, direction, sight_herding):
    """Compute herding using block list"""
    index_list, count, offset, shape = block_list(position, sight_herding)
    return herding_interaction(position, direction, sight_herding, index_list, count,
                               offset, shape)
