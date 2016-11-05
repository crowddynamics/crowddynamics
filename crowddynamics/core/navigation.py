from collections import Iterable

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numba
import numpy as np
import skfmm
import skimage.draw
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from typing import Optional

from crowddynamics.core.geometry import shapes_to_point_pairs


def _to_indices(points, step):
    """

    Args:
        points (numpy.ndarray): Points on a continuous grid
        step (float): Step size of the grid

    Returns:
        Array of integers. Indices of a point in discrete grid.

    """
    return np.round(points / step).astype(np.int64)


def _set_values(grid, step, shape, value):
    """
    Set values on discrete grid using ``scikit-image``.

    Args:
        shape (BaseGeometry): Shapely shape
        grid (numpy.ndarray): Grid to set values
        value (grid.dtype): Value to set to the grid points
        step (float): Step size of the grid

    Returns:
        None. Values are set to grid.

    """
    if isinstance(shape, Polygon):
        points = np.asarray(shape.exterior)
        points = _to_indices(points, step)
        x, y = points[:, 0], points[:, 1]
        j, i = skimage.draw.polygon(x, y)
        grid[i, j] = value
    elif isinstance(shape, LineString):
        points = shapes_to_point_pairs(shape)
        points = _to_indices(points, step)
        for args in points:
            j, i = skimage.draw.line(*args.flatten())
            grid[i, j] = value
    elif isinstance(shape, Iterable):
        for shape_ in shape:
            _set_values(grid, step, shape_, value)
    elif shape is None:
        # If shape is None do nothing.
        pass
    else:
        raise Exception("Type of shape is not valid.")


def distance_map(step: float, domain: Polygon,
                 targets: Optional[LineString] = None,
                 obstacles: Optional[LineString] = None):
    """
    Solve distance map using Fast Marching Method (FMM) from ``scikit-fmm``.

    Args:
        domain (Polygon): Domain containing obstacles and targets
        obstacles (LineString, optional): Impassable regions in the domain
        targets (LineString, optional): Target regions in the domain
        step (float): Step size for the meshgrid.

    Return:
        (numpy.meshgrid, numpy.ndarray, numpy.ma.MaskedArray):
            (mgrid, dmap, phi)

    """
    # Numerical values for objects in the domain
    initial_value = -1.0  # Empty regions
    target_value = 1.0  # Target regions
    obstacle_value = True  # Obstacle regions

    # Form the meshgrid
    minx, miny, maxx, maxy = domain.bounds  # Bounding box
    x = np.arange(minx, maxx + step, step=step)  # x-axis
    y = np.arange(miny, maxy + step, step=step)  # y-axis
    mgrid = np.meshgrid(x, y)  # (X, Y)

    # Contour used for solving distance map
    # Mask for masked values that represent obstacles for the solver
    contour = np.full_like(mgrid[0], initial_value, dtype=np.float64)
    mask = np.full_like(mgrid[0], False, dtype=np.bool_)

    _set_values(contour, step, targets, target_value)
    _set_values(mask, step, obstacles, obstacle_value)

    phi = np.ma.MaskedArray(contour, mask)

    # Solve distance map using Fast-Marching Method (FMM)
    dmap = skfmm.distance(phi, dx=step)

    return mgrid, dmap, phi


def direction_map(dmap):
    """
    Normalized gradient of distance map.

    Args:
        dmap (numpy.ndarray): Distance map.

    Returns:
        numpy.ndarray: Direction map. Array of shape: ``dmap.shape + (2,)``

    """
    u, v = np.gradient(dmap)
    dir_map = np.zeros(u.shape + (2,))

    # Flip order from (row, col) to (x, y)
    l = np.hypot(u, v)
    # TODO: handle l == 0
    dir_map[:, :, 0] = v / l
    dir_map[:, :, 1] = u / l
    return dir_map


def plot_distance_map(mgrid, dmap, phi):
    """
    Plot distance map

    Args:
        mgrid (numpy.meshgrid):
        dmap (numpy.ndarray):
        phi (numpy.ma.MaskedArray):

    """
    X, Y = mgrid
    bbox = (X.min(), X.max(), Y.min(), Y.max())

    opts = dict(
        figsize=(12, 12),
    )

    fig, ax = plt.subplots(**opts)

    # Distance map plot
    ax.imshow(dmap, interpolation='bilinear', origin='lower', cmap=cm.gray,
              extent=bbox)
    ax.contour(X, Y, dmap, 30, linewidths=1, colors='gray')  # Contour lines
    ax.contour(X, Y, phi.mask, [0], linewidths=1, colors='black')  # Obstacles

    # plt.savefig("distance_map_{}.pdf".format(name))
    ax.show()


@numba.jit(nopython=True)
def merge_dir_maps(dmap, dir_map1, dir_map2, radius, value):
    """

    Args:
        dmap:
        dir_map1:
        dir_map2:
        radius (float): Radius
        value (float): Value between (0, 1). Value denoting the strength of
            dir_map1 at distance of radius.

    Returns:
        numpy.ndarray:

    """
    merged = np.copy(dir_map2)

    n, m = dmap.shape
    for i in range(n):
        for j in range(m):
            x = np.abs(dmap[i, j])

            # Decreasing function
            # f(x) = value ** (x / radius)
            if x < 2 * radius:
                k = value ** (x / radius)
                merged[i, j] = - k * dir_map1[i, j] + (1 - k) * dir_map2[i, j]
            else:
                # Does not change value
                pass

    return merged


def static_potential(step: float, domain: Polygon,
                     exits: Optional[LineString] = None,
                     obstacles: Optional[LineString] = None,
                     radius=0.3,
                     value=0.1):
    """

    Args:
        step:
        domain:
        exits:
        obstacles:
        value:
        radius:

    Returns:

    """
    _, dmap_exits, _ = distance_map(step, domain, exits, obstacles)
    _, dmap_obs, _ = distance_map(step, domain, obstacles, None)

    dir_map_exits = direction_map(dmap_exits)
    dir_map_obs = direction_map(dmap_obs)

    dir_map = merge_dir_maps(dmap_obs, dir_map_obs, dir_map_exits, radius, value)

    return dir_map
