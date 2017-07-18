r"""
Navigation
----------
Navigation aka path-planing algorithms computes target direction
:math:`\mathbf{\hat{e}_0}` for each agent. Target direction accounts for agents
desire to move towards some direction.

Navigation algorithms can be combined. For examples `static potential` is
combination of `shortest path` and `obstacle handling`.
"""
from typing import Tuple

import numba
import numpy as np
from numba import f8, i8
from shapely.geometry import Polygon

from crowddynamics.core.steering.collective_motion import leader_follower_with_herding_interaction
from crowddynamics.core.steering.obstacle_handling import \
    direction_map_obstacles, obstacle_handling
from crowddynamics.core.steering.quickest_path import MeshGrid, DistanceMap, \
    DirectionMap, meshgrid, shortest_path


def static_potential(domain, targets, obstacles,
                     step: float, radius: float, strength: float) -> \
        Tuple[MeshGrid, DistanceMap, DirectionMap]:
    r"""
    Navigation algorithm that uses combines `shortest path` and
    `obstacle handling` into single navigation mesh that guides agents towards
    targets so that they don't willingly collide with obstacles (they can be
    pushed towards obstacles by other agents).

    Args:
        step (float):
        domain (Polygon):
        targets (LineString, optional):
        obstacles (LineString, optional):
        strength (float):
        radius (float):

    Returns:
        Tuple[MeshGrid, DirectionMap]:
    """
    # Compute meshgrid for solving distance maps.
    mgrid = meshgrid(step, *domain.bounds)

    dir_map_targets, dmap_targets = shortest_path(mgrid, domain, targets,
                                                  obstacles, radius)
    dir_map_obs, dmap_obs = direction_map_obstacles(mgrid, obstacles)

    # Combines two direction maps in a way that agents do not run into a wall
    dir_map = obstacle_handling(dmap_obs, dir_map_obs, dir_map_targets, radius,
                                strength)

    return mgrid, dmap_targets, dir_map


@numba.jit(nopython=True, nogil=True, cache=True)
def is_inside(a, lower, upper):
    for l, i, u in zip(lower, a, upper):
        if not (l <= i < u):
            return False
    return True


@numba.jit((i8[:, :], numba.types.Tuple((f8[:, :], f8[:, :])), f8[:, :]),
           nopython=True, nogil=True, cache=True)
def getdefault(indices, dir_map, defaults):
    assert indices.shape == defaults.shape
    out = np.copy(defaults)
    x, y = dir_map
    for k in range(len(indices)):
        i, j = indices[k]
        if is_inside((i, j), (0, 0), x.shape):
            out[k][0] = x[i, j]
            out[k][1] = y[i, j]
    return out
