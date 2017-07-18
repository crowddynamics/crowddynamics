"""Collective motion"""
import numba
import numpy as np
from cell_lists import add_to_cells, neighboring_cells, iter_nearest_neighbors
from numba import f8, i8
from numba.typing.typeof import typeof

from crowddynamics.core.sensory_region import is_obstacle_between_points
from crowddynamics.core.structures import obstacle_type_linear
from crowddynamics.core.vector2D import length, normalize, weighted_average, dot
from crowddynamics.simulation.agents import NO_LEADER, NO_TARGET

MISSING_NEIGHBOR = -1


@numba.jit([f8[:, :](f8[:, :])], nopython=True, nogil=True, cache=True)
def normalize_nx2(v):
    out = np.zeros_like(v)
    for i in range(len(v)):
        out[i, :] = normalize(v[i, :])
    return out


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


@numba.jit(nopython=True, nogil=True)
def set_neighbor(i, j, l, neighbors, distances, distances_max):
    argmax = np.argmax(distances[i, :])
    neighbors[i, argmax] = j
    distances[i, argmax] = l
    distances_max[i] = np.max(distances[i, :])


@numba.jit([(f8[:, :], f8, i8, i8[:], i8[:], i8[:],
             i8[:], i8[:], typeof(obstacle_type_linear)[:])],
           nopython=True, nogil=True, cache=True)
def find_nearest_neighbors(
        position, sight, size_nearest_other,
        cell_indices, neigh_cells, points_indices, cells_count,
        cells_offset, obstacles):
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

    for i, j in iter_nearest_neighbors(
            cell_indices, neigh_cells, points_indices, cells_count,
            cells_offset):
        # Test if line of sight is obstructed by an obstacle
        if is_obstacle_between_points(position[i], position[j], obstacles):
            continue

        l = length(position[i] - position[j])

        if l < distances_max[i]:
            set_neighbor(i, j, l, neighbors, distances, distances_max)

        if l < distances_max[j]:
            set_neighbor(j, i, l, neighbors, distances, distances_max)

    return neighbors


@numba.jit(nopython=True, nogil=True, cache=True)
def herding_interaction(is_herding, position, velocity, neighbors,
                        weight_position, phi):
    r"""Herding effect.

        .. math::
           \mathbf{\hat{e}_{herding}} = \mathcal{N}
           \left(\sum_{j \in Neigh} \mathbf{\hat{e}}_j\right)

    """
    # [(boolean[:], f8[:, :], f8[:, :], i8[:, :], f8, f8, i8[:])]
    new_direction = np.zeros_like(position)
    has_new_direction = np.zeros(position.shape[0], dtype=np.bool_)

    mean_position = np.zeros(position.shape[1])
    mean_velocity = np.zeros(velocity.shape[1])

    for i in range(position.shape[0]):
        if not is_herding[i]:
            continue

        num_neigh = 0

        for j in neighbors[i, :]:
            if j == MISSING_NEIGHBOR:
                continue

            is_heading_away, _ = herding_relationship(
                position[i], position[j], velocity[i], velocity[j], phi)

            if is_heading_away:
                mean_position[:] += position[j, :]
                mean_velocity[:] += velocity[j, :]
                num_neigh += 1

        if num_neigh > 0:
            new_direction[i, :] = normalize(weighted_average(
                normalize(mean_position[:] / num_neigh - position[i, :]),
                normalize(mean_velocity[:]),
                weight_position))
            has_new_direction[i] = True

        mean_position[:] = 0
        mean_velocity[:] = 0

    return new_direction, has_new_direction


@numba.jit(nopython=True, nogil=True, cache=True)
def leader_follower_interaction_brute(
        is_follower, is_leader, position, velocity, weight_position, phi,
        target, index_leader, obstacles, sight):
    has_strategy = np.zeros(len(position), dtype=np.bool_)
    new_direction = np.zeros_like(position)
    # new_target = np.zeros_like(target)
    # new_index_leader = np.zeros_like(index_leader)

    indices = np.arange(len(position))
    leaders = indices[is_leader]
    distances = np.zeros(leaders.shape, dtype=np.float64)

    # Iterate over followers
    for i in indices[is_follower]:
        behind_obstacle = 0
        heading_away = 0

        # Find distances to the leaders
        for k, j in enumerate(leaders):
            distances[k] = length(position[i] - position[j])

        # Iterate over leaders in orders from closest to furthest leader
        # and choose a strategy.
        for k in np.argsort(distances):
            if distances[k] > sight:
                continue

            j = leaders[k]

            if is_obstacle_between_points(position[i], position[j], obstacles):
                # We are not seeing this leader.
                leader = index_leader[i]
                # Check if we were following this leader before.
                if leader != NO_LEADER and leader == j:
                    behind_obstacle += 1
                    # Use navigation of this leader.
                    target[i] = target[leader]
                    has_strategy[i] = True
                    break
                else:
                    continue
            else:
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
                    has_strategy[i] = True
                    break

        if behind_obstacle == 0 and heading_away == 0:
            leader = index_leader[i]
            if leader != NO_LEADER:
                target[i] = target[leader]
                has_strategy[i] = True

    return new_direction, has_strategy


def leader_follower_interaction(
        agents, obstacles, sight, phi=0.45 * np.pi,
        weight_position_leader=0.40):
    # Follow the leader
    direction_leader, has_strategy = leader_follower_interaction_brute(
        agents['is_follower'], agents['is_leader'], agents['position'],
        agents['velocity'], weight_position_leader, phi,
        agents['target'], agents['index_leader'], obstacles, sight)

    # Use familiar exits
    is_lost = ~has_strategy & agents['is_follower']
    agents['target'][is_lost] = agents['familiar_exit'][is_lost]

    return direction_leader


def leader_follower_with_herding_interaction(
        agents, obstacles, sight,
        size_nearest_other,
        phi=0.45 * np.pi,
        weight_position_herding=0.15,
        weight_position_leader=0.40,
        weight_direction_leader=0.65):
    position = agents['position']
    velocity = agents['velocity']
    is_leader = agents['is_leader']
    is_follower = agents['is_follower']
    sight_leader = 20.0

    cell_size = sight
    points_indices, cells_count, cells_offset, grid_shape = add_to_cells(
        agents['position'], cell_size)
    cell_indices = np.arange(len(cells_count))
    neigh_cells = neighboring_cells(grid_shape)

    neighbors = find_nearest_neighbors(
        position, sight,
        size_nearest_other, cell_indices, neigh_cells, points_indices,
        cells_count, cells_offset, obstacles)

    direction, has_direction = herding_interaction(
        is_follower, position, velocity, neighbors, weight_position_herding,
        phi)

    agents['target'][has_direction] = NO_TARGET

    direction_leader, has_strategy = leader_follower_interaction_brute(
        is_follower, is_leader, position, velocity, weight_position_leader,
        phi, agents['target'], agents['index_leader'], obstacles, sight_leader)

    # Use familiar exits
    is_lost = ~(has_direction | has_strategy) & is_follower
    agents['target'][is_lost] = agents['familiar_exit'][is_lost]

    return normalize_nx2(weighted_average(
        direction_leader, direction, weight_direction_leader))
