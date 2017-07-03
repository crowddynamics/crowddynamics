"""Herding / Flocking / Leader-Follower effect"""
import numba
import numpy as np
from numba import f8, i8, boolean
from numba.types import optional
from numba.typing.typeof import typeof

from crowddynamics.core.geom2D import line_intersect
from crowddynamics.core.interactions.block_list import block_list, get_block
from crowddynamics.core.sensory_region import is_obstacle_between_points
from crowddynamics.core.structures import obstacle_type_linear
from crowddynamics.core.vector2D import length, normalize, weighted_average, dot

# FIXME:
from crowddynamics.simulation.agents import NO_LEADER, NO_TARGET


@numba.jit([(f8[:], f8[:], f8[:], f8[:], f8)],
           nopython=True, nogil=True, cache=True)
def herding_relationship(x1, x2, v1, v2, phi=np.pi / 2):
    """Determine leader-follower relationship

    Args:
        x1:
        x2:
        v1:
        v2:
        phi: Angle between [0, pi]

    Returns:
        (bool ,bool):
    """
    if length(v1) == 0 or length(v2) == 0:
        return False, False

    e_rel = normalize(x2 - x1)

    c_i = dot(e_rel, normalize(v1))
    c_j = -dot(e_rel, normalize(v2))

    # 1.0 = cos(0)
    cos_phi = np.cos(phi)
    if cos_phi < c_i < 1.0:
        if cos_phi < c_j < 1.0:
            return False, False
        else:
            return True, False
    else:
        if cos_phi < c_j < 1.0:
            return False, True
        else:
            return True, True


MISSING_NEIGHBOR = -1
NEIGHBOR_INDICES = ((1, 0), (1, 1), (0, 1), (1, -1))


@numba.jit([f8[:, :](f8[:, :])], nopython=True, nogil=True, cache=True)
def normalize_nx2(v):
    out = np.zeros_like(v)
    for i in range(len(v)):
        out[i, :] = normalize(v[i, :])
    return out


@numba.jit([(i8[:], i8[:], i8[:], i8[:])], nopython=True, nogil=True)
def iter_block_list(index_list, count, offset, shape):
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


@numba.jit([(optional(boolean[:]), f8[:, :], f8[:, :], f8, i8, i8[:], i8[:],
             i8[:], i8[:],
             typeof(obstacle_type_linear)[:])],
           nopython=True, nogil=True)
def find_nearest_neighbors(followable, position, velocity, sight,
                           neighborhood_size,
                           index_list, count, offset, shape, obstacles):
    size = len(position)

    neighbors = np.full((size, neighborhood_size),
                        fill_value=MISSING_NEIGHBOR,
                        dtype=np.int64)
    '''Current nearest neighbours.'''

    distances = np.full((size, neighborhood_size),
                        fill_value=np.inf,
                        dtype=np.float64)
    '''Distance to current nearest neighbours.'''

    distances_max = np.full(size,
                            fill_value=np.inf,
                            dtype=np.float64)
    '''Distance to furthest neighbor.'''

    for i, j in iter_block_list(index_list, count, offset, shape):
        l = length(position[i] - position[j])

        if l < sight:
            # Test if line of sight is obstructed by an obstacle
            obstructed = False
            for w in range(len(obstacles)):
                if line_intersect(obstacles[w]['p0'], obstacles[w]['p1'],
                                  position[i], position[j]):
                    obstructed = True
                    break

            if obstructed:
                continue

            # b1, b2 = follower_relationship(
            #     position[i], position[j], velocity[i], velocity[j])

            if l < distances_max[i] and followable is not None and followable[
                j]:
                argmax = np.argmax(distances[i, :])
                neighbors[i, argmax] = j
                distances[i, argmax] = l
                distances_max[i] = np.max(distances[i, :])

            if l < distances_max[j] and followable is not None and followable[
                i]:
                argmax = np.argmax(distances[j, :])
                neighbors[j, argmax] = i
                distances[j, argmax] = l
                distances_max[j] = np.max(distances[j, :])

    return neighbors


@numba.jit(nopython=True, nogil=True)
def set_neighbor(i, j, l, neighbors, distances, distances_max):
    argmax = np.argmax(distances[i, :])
    neighbors[i, argmax] = j
    distances[i, argmax] = l
    distances_max[i] = np.max(distances[i, :])


@numba.jit([(boolean[:], f8[:, :], f8, i8, i8, i8[:], i8[:], i8[:],
             i8[:], typeof(obstacle_type_linear)[:])],
           nopython=True, nogil=True)
def find_nearest_neighbors_with_leaders(
        is_leader, position,
        sight, size_nearest_leaders, size_nearest_other,
        index_list, count, offset, shape, obstacles):
    size = len(position)

    neighbors = np.full(
        (size, size_nearest_other),
        fill_value=MISSING_NEIGHBOR, dtype=np.int64)
    '''Current nearest neighbours.'''
    distances = np.full(
        (size, size_nearest_other),
        fill_value=sight, dtype=np.float64)
    '''Distance to current nearest neighbours.'''
    distances_max = np.full(
        size, fill_value=sight, dtype=np.float64)
    '''Distance to furthest neighbor.'''

    neighbors_leaders = np.full(
        (size, size_nearest_leaders),
        fill_value=MISSING_NEIGHBOR, dtype=np.int64)
    '''Current nearest neighbours.'''
    distances_leaders = np.full(
        (size, size_nearest_leaders),
        fill_value=sight, dtype=np.float64)
    '''Distance to current nearest neighbours.'''
    distances_max_leader = np.full(
        size, fill_value=sight, dtype=np.float64)
    '''Distance to furthest neighbor.'''

    for i, j in iter_block_list(index_list, count, offset, shape):
        l = length(position[i] - position[j])

        if l < distances_max_leader[i] and is_leader[j]:
            set_neighbor(i, j, l, neighbors_leaders, distances_leaders,
                         distances_max_leader)

        if l < distances_max_leader[j] and is_leader[i]:
            set_neighbor(j, i, l, neighbors_leaders, distances_leaders,
                         distances_max_leader)

        # Test if line of sight is obstructed by an obstacle
        if is_obstacle_between_points(position[i], position[j], obstacles):
            continue

        if l < distances_max_leader[i]:
            set_neighbor(i, j, l, neighbors, distances, distances_max)

        if l < distances_max_leader[j]:
            set_neighbor(j, i, l, neighbors, distances, distances_max)

    return neighbors, neighbors_leaders


@numba.jit([f8[:, :](boolean[:], f8[:, :], f8[:, :], i8[:, :], f8, f8)],
           nopython=True, nogil=True, cache=True)
def herding_interaction(is_herding, position, velocity, neighbors,
                        weight_position, phi):
    r"""Herding effect.

    .. math::
       \mathbf{\hat{e}_{herding}} = \mathcal{N}
       \left(\sum_{j \in Neigh} \mathbf{\hat{e}}_j\right)

    Args:
        position:
        is_herding:
        weight_position:
        velocity:
        neighbors:

    Returns:
        numpy.ndarray:
            New direction vector :math:`\mathbf{\hat{e}_{herding}}`
    """
    new_direction = np.zeros_like(position)
    mean_position = np.zeros(position.shape[1])
    mean_velocity = np.zeros(velocity.shape[1])

    m, n = neighbors.shape
    for i in range(m):
        if not is_herding[i]:
            # i is not herding agent
            continue

        num_neigh = 0

        for row in range(n):
            j = neighbors[i, row]
            if j == MISSING_NEIGHBOR:
                continue

            is_following, _ = herding_relationship(
                position[i], position[j], velocity[i], velocity[j], phi)

            if is_following:
                mean_position[:] += position[j, :]
                mean_velocity[:] += velocity[j, :]
                num_neigh += 1

        if num_neigh > 0:
            new_direction[i, :] = normalize(weighted_average(
                normalize(mean_position[:] / num_neigh - position[i, :]),
                normalize(mean_velocity[:]),
                weight_position))

        mean_position[:] = 0
        mean_velocity[:] = 0

    return new_direction


def weight_function(c_j, weight_position):
    if c_j < 0:
        return weight_position
    else:
        return weight_position + (-3 * weight_position + 3) * c_j ** 2 + \
               (2 * weight_position - 2) * c_j ** 3


def leader_follower_interaction(is_follower, position, velocity, neighbors,
                                weight_position, phi, target, index_leader,
                                obstacles):
    """Leader follower interaction

    Args:
        is_follower:
        position:
        velocity:
        neighbors:
        weight_position:
        phi:
        target:
        index_leader:
        obstacles:

    Returns:

    """
    new_direction = np.zeros_like(position)

    m, n = neighbors.shape
    for i in range(m):
        if not is_follower[i]:
            continue  # i is not herding agent

        behind_obstacle = 0
        heading_away = 0
        for row in range(n):
            j = neighbors[i, row]
            if j == MISSING_NEIGHBOR:
                continue

            if is_obstacle_between_points(position[i], position[j], obstacles):
                # We are not seeing this leader.
                leader = index_leader[i]
                # Check if we were following this leader before.
                if leader != NO_LEADER and leader == j:
                    behind_obstacle += 1
                    # Use navigation of this leader.
                    target[i] = target[leader]
                    break

                continue

            is_heading_away, _ = herding_relationship(
                position[i], position[j], velocity[i], velocity[j], phi)

            if is_heading_away:
                heading_away += 1
                # We see this leader. Remember it and follow.
                index_leader[i] = j
                target[i] = NO_TARGET
                new_direction[i, :] = normalize(weighted_average(
                    normalize(position[j, :] - position[i, :]),
                    normalize(velocity[j, :]), weight_position))
                break

        if behind_obstacle == 0 and heading_away == 0:
            leader = index_leader[i]
            if leader != NO_LEADER:
                target[i] = target[leader]

    return new_direction


def herding_block_list(agents, obstacles, sight,
                       size_nearest_leaders,
                       size_nearest_other,
                       phi=0.45 * np.pi,
                       weight_position_herding=0.15,
                       weight_position_leader=0.40,
                       weight_direction_leader=0.65):
    position = agents['position']
    velocity = agents['velocity']
    is_leader = agents['is_leader']
    is_follower = agents['is_follower']

    # Indices of all agents
    indices = np.arange(len(position))

    index_list, count, offset, shape = block_list(position, sight)
    neighbors, neighbors_leaders = find_nearest_neighbors_with_leaders(
        is_leader, position, sight, size_nearest_leaders,
        size_nearest_other, index_list, count, offset, shape, obstacles)

    direction = herding_interaction(
        is_follower, position, velocity, neighbors, weight_position_herding,
        phi)

    direction_leader = leader_follower_interaction(
        is_follower, position, velocity, neighbors_leaders,
        weight_position_leader, phi, agents['target'], agents['index_leader'],
        obstacles)

    return normalize_nx2(weighted_average(
        direction_leader, direction, weight_direction_leader))
