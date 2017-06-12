"""Herding / Flocking / Leader-Follower effect"""
import numba
import numpy as np
from numba import f8, i8
from numba.typing.typeof import typeof

from crowddynamics.core.geom2D import line_intersect
from crowddynamics.core.interactions.block_list import block_list, get_block
from crowddynamics.core.structures import obstacle_type_linear
from crowddynamics.core.vector2D import length, normalize

EMPTY = -1
NEIGHBOR_INDICES = ((1, 0), (1, 1), (0, 1), (1, -1))


@numba.jit([(i8[:], i8[:], i8[:], i8[:])], nopython=True, nogil=True)
def block_list_iter(index_list, count, offset, shape):
    r"""Iterate over blocklist

    Args:
        index_list:
        count:
        offset:
        shape:

    Returns:
        numpy.ndarray:
            New direction vector :math:`\mathbf{\hat{e}_{herding}}`
    """
    n, m = shape

    for i in range(n):
        for j in range(m):
            # Herding between agents inside the block
            ilist = get_block((i, j), index_list, count, offset, shape)
            for l, i_agent in enumerate(ilist[:-1]):
                for j_agent in ilist[l + 1:]:
                    yield i_agent, j_agent

            # Herding between agent inside the block and neighbouring agents
            for k in range(len(NEIGHBOR_INDICES)):
                i2, j2 = NEIGHBOR_INDICES[k]
                if 0 <= (i + i2) < n and 0 <= (j + j2) < m:
                    ilist2 = get_block((i + i2, j + j2), index_list, count,
                                       offset, shape)
                    for i_agent in ilist:
                        for j_agent in ilist2:
                            yield i_agent, j_agent


@numba.jit([(f8[:, :], f8, i8, i8[:], i8[:], i8[:], i8[:],
             typeof(obstacle_type_linear)[:])],
           nopython=True, nogil=True)
def compute_neighbors(position, sight, neighborhood_size,
                      index_list, count, offset, shape,
                      obstacles):
    agent_size = len(position)

    neighbors = np.full((agent_size, neighborhood_size),
                        fill_value=EMPTY,
                        dtype=np.int64)
    '''Current nearest neighbours.'''
    distances = np.full((agent_size, neighborhood_size),
                        fill_value=np.inf,
                        dtype=np.float64)
    '''Distance to current nearest neighbours.'''
    distances_max = np.full(agent_size,
                            fill_value=np.inf,
                            dtype=np.float64)
    '''Distance to furthest neighbor.'''

    for i, j in block_list_iter(index_list, count, offset, shape):
        l = length(position[i] - position[j])

        # Test if line of sight is obstructed by an obstacle
        obstructed = False
        for w in range(len(obstacles)):
            if line_intersect(obstacles[w]['p0'], obstacles[w]['p1'],
                              position[i], position[j]):
                obstructed = True
                break

        if obstructed:
            continue

        if l < sight:
            if l < distances_max[i]:
                argmax = np.argmax(distances[i, :])
                neighbors[i, argmax] = j
                distances[i, argmax] = l
                distances_max[i] = np.max(distances[i, :])

            if l < distances_max[j]:
                argmax = np.argmax(distances[j, :])
                neighbors[j, argmax] = i
                distances[j, argmax] = l
                distances_max[j] = np.max(distances[j, :])

    return neighbors


@numba.jit([f8[:, :](f8[:, :], i8[:, :])], nopython=True, nogil=True, cache=True)
def herding_interaction(velocity, neighbors):
    r"""Herding effect.

    .. math::
       \mathbf{\hat{e}_{herding}} = \mathcal{N}
       \left(\sum_{j \in Neigh} \mathbf{\hat{e}}_j\right)

    Args:
        velocity:
        neighbors:

    Returns:
        numpy.ndarray:
            New direction vector :math:`\mathbf{\hat{e}_{herding}}`
    """
    new_direction = np.zeros_like(velocity)
    n, m = neighbors.shape
    for i in range(n):
        for row in range(m):
            j = neighbors[i, row]
            if j == EMPTY:
                continue
            new_direction[i, :] += velocity[j, :]
        new_direction[i, :] = normalize(new_direction[i, :])
    return new_direction


def herding_block_list(position, velocity, sight, neighborhood_size, obstacles):
    index_list, count, offset, shape = block_list(position, sight)
    neighbors = compute_neighbors(position, sight, neighborhood_size,
                                  index_list, count, offset, shape, obstacles)
    return herding_interaction(velocity, neighbors)
