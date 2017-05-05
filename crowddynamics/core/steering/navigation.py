r"""
Navigation
----------
Navigation aka path-planing algorithms computes target direction 
:math:`\mathbf{\hat{e}_0}` for each agent. Target direction accounts for agents 
desire to move towards some direction.

Currently implemented navigation 

- Quickest path
- Obstacle handling
- Herding / Leader Follower

"""
from typing import Tuple

import numba
import numpy as np
from loggingtools import log_with
from numba import f8, i8
from shapely.geometry import Polygon

from crowddynamics.core.steering.obstacle_handling import \
    direction_map_obstacles, obstacle_handling
from crowddynamics.core.steering.quickest_path import MeshGrid, DistanceMap, \
    DirectionMap, meshgrid, direction_map_targets


@log_with(timed=True)
def static_potential(domain,
                     targets,
                     obstacles,
                     step: float,
                     radius: float,
                     strength: float) -> \
        Tuple[MeshGrid, DistanceMap, DirectionMap]:
    r"""Static potential

    Navigation algorithm that does not take into account
    the space that is occupied by dynamic agents (aka agents).

    1. Discretize the domain into grid
    2. Vector field pointing to target
    
        a. Solve distance map from the targets using buffered obstacles
        b. Compute gradient of the distance map to obtain the vector field
        c. Fill the missing values by interpolating nearest values

    3. Vector field pointing to obstacles
    
        a. Compute distance map from the obstacles
        b. Compute gradient of the distance map to obtain the vector field
    
    4. Combine these two vector fields

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

    dir_map_targets, dmap_targets = direction_map_targets(
        mgrid, domain, targets, obstacles, radius)
    dir_map_obs, dmap_obs = direction_map_obstacles(mgrid, obstacles)

    # Combines two direction maps in a way that agents do not run into a wall
    dir_map = obstacle_handling(dmap_obs, dir_map_obs, dir_map_targets, radius,
                                strength)

    return mgrid, dmap_targets, dir_map


@numba.jit()
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


@numba.jit()
def set_target_direction(array, indices, values):
    for j, i in enumerate(indices):
        array[i]['target_direction'][:] = values[j, :]


def navigation(agents, mask, mgrid, dir_map):
    """Find directions for positions

    Args:
        position (numpy.ndarray): Shape (n, 2)
        step (float):
        dir_map (Tuple[numpy.ndarray, numpy.ndarray]):

    Returns:
        numpy.ndarray:

    References:
        - https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    """
    # Flip x and y to array index i and j
    indices = np.fliplr(mgrid.indicer(agents[mask]['position']))
    new_direction = getdefault(indices, dir_map, agents[mask]['target_direction'])
    set_target_direction(agents, np.arange(len(agents))[mask], new_direction)
