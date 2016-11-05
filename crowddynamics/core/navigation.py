from collections import Iterable

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skfmm
import skimage.draw
from shapely.geometry import LineString, Polygon

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
        shape: Continuous shape (Polygon, LineString, ...)
        grid: Grid to set values
        value: Value to set to the grid points
        step: Step size of the grid

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
    else:
        raise Exception("Type of shape is not valid.")


def distance_map(domain: Polygon,
                 obstacles: LineString,
                 targets: LineString,
                 step: float):
    """
    Solve distance map using Fast Marching Method (FMM) from ``scikit-fmm``.

    Args:
        domain (Polygon): Domain containing obstacles and targets
        obstacles (LineString): Impassable regions in the domain
        targets (LineString): Target regions in the domain
        step (float): Step size for the meshgrid.

    Return:
        -

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
    Gradient of direction map. Normalization of the gradient is not required
    because the distance map is already normal due to the initial values
    used to solve the distance map.

    Returns:
        Direction Map

    """
    u, v = np.gradient(dmap)
    dir_map = np.zeros(u.shape + (2,))
    l = np.hypot(u, v)

    # Flip order from (row, col) to (x, y)
    dir_map[:, :, 0] = v / l
    dir_map[:, :, 1] = u / l
    return dir_map


def plot_distance_map(mgrid, dmap, phi):
    """
    Plot distance map

    Args:
        mgrid (numpy.meshgrid):
        dmap (numpy.ndarray):
        phi np.ma.MaskedArray:

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
