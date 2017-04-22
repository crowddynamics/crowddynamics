"""Navigation/path-planning algorithms

Continuos shortest path problem

Eikonal equation solvers

- Fast Marching Method [scikit-fmm]
- Fast Iterative Method [SCI-Solver_Eikonal]

References:
    .. [scikit-fmm] https://github.com/scikit-fmm/scikit-fmm
    .. [SCI-Solver_Eikonal] https://github.com/SCIInstitute/SCI-Solver_Eikonal

Attributes:
    MeshGrid (namedtuple):
        - values: Tuple (X, Y) of ndarrays of shape (n, m).
        - shape:
        - step: 
        - bounds:
        - indicer:
    DistanceMap:
        Ndarray of shape (n, m) where the values indicate the shortest distance
        from target.
    DirectionMap:
        Tuple (U, V) of ndarray of shape (n, m), where the value are x and y
        components of a (unit)vector field.
"""
from collections import namedtuple
from typing import Tuple, Optional

import numba
import numpy as np
import skfmm
from loggingtools.log_with import log_with
from numba import f8, i8
from scipy.interpolate import NearestNDInterpolator
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from skimage.segmentation import find_boundaries

from crowddynamics.core.geometry import geom_to_skimage

MeshGrid = namedtuple('MeshGrid', 'values shape step bounds indicer')
DistanceMap = np.ma.MaskedArray
DirectionMap = Tuple[np.ma.MaskedArray, np.ma.MaskedArray]


# Numerical Routines

@log_with(arguments=False, timed=True)
def fill_missing(mgrid: MeshGrid, dir_map: DirectionMap):
    """Fill missing value with by interpolating the values from nearest neighbours

    Args:
        mgrid:
        dir_map:

    Returns:
        DirectionMap:
    """
    x, y = mgrid.values
    u, v = dir_map

    # Construct the interpolators from the boundary values surrounding the
    # missing values
    boundaries = find_boundaries(u.mask, mode='outer')
    points = np.stack((y[boundaries], x[boundaries])).T
    ip_u = NearestNDInterpolator(points, u[boundaries], rescale=False)
    ip_v = NearestNDInterpolator(points, v[boundaries], rescale=False)

    # interpolate only missing values (u.mask)
    missing = (y[u.mask], x[v.mask])
    u[u.mask] = ip_u(missing)
    v[v.mask] = ip_v(missing)


@log_with(arguments=False, timed=True)
@numba.jit((f8[:, :], numba.types.Tuple((f8[:, :], f8[:, :])),
            numba.types.Tuple((f8[:, :], f8[:, :])), f8, f8),
           nopython=True, nogil=True, cache=True)
def merge_dir_maps(dmap, dir_map_obs, dir_map_targets, radius, strength):
    r"""Merges direction maps

    Function that merges two direction maps together. Let distance map from
    obstacles be :math:`\Phi(\mathbf{x})` and :math:`\lambda(\Phi(\mathbf{x}))`
    be any decreasing function :math:`\frac{\partial}{\partial\Phi}\lambda(\Phi(\mathbf{x})) < 0` of
    distance from obstacles such that

    .. math::
       \lambda(\Phi) &=
       \begin{cases}
       1 & \Phi = 0 \\
       0 & \Phi > M > 0
       \end{cases}

    Then merged direction map :math:`\hat{\mathbf{e}}_{merged}` is

    .. math::
       k &= \lambda(\Phi(\mathbf{x})) \\
       \hat{\mathbf{e}}_{merged} &= k \hat{\mathbf{e}}_{obs} + (1 - k) \hat{\mathbf{e}}_{exits}

    Args:
        dmap:
        dir_map_obs:
        dir_map_targets:
        radius (float):
            Radius
        strength (float):
            Value between (0, 1). Value denoting the strength of dir_map1 at
            distance of radius.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
    """
    # FIXME: artifacts near radius distance from obstacles
    u1, v1 = dir_map_obs
    u2, v2 = dir_map_targets
    u_out, v_out = np.copy(u2), np.copy(v2)

    n, m = dmap.shape
    for i in range(n):
        for j in range(m):
            # Distance from the obstacles
            x = np.abs(dmap[i, j])
            if 0 < x < radius:
                # Decreasing function
                k = strength ** (x / radius)
                u_out[i, j] = - k * u1[i, j] + (1 - k) * u2[i, j]
                v_out[i, j] = - k * v1[i, j] + (1 - k) * v2[i, j]

    # Normalize the output
    l = np.hypot(u_out, v_out)
    return u_out / l, v_out / l


# Grid

@log_with(arguments=False, timed=True)
def meshgrid(step: float, minx: float, miny: float,
             maxx: float, maxy: float) -> MeshGrid:
    """2-Dimensional meshgrid with inclusive end points maxx and maxy

    Args:
        step (float): 
        minx (float): 
        miny (float): 
        maxx (float): 
        maxy (float):

    Returns:
        MeshGrid: 
    """
    x = np.arange(minx, maxx + step, step=step)
    y = np.arange(miny, maxy + step, step=step)
    values = np.meshgrid(x, y, indexing='xy')
    shape = values[0].shape

    def indicer(position):
        """Converts positions to meshgrid indices"""
        shifted = np.asarray(position) - np.array((minx, miny))
        return (shifted / step).astype(np.int64)

    return MeshGrid(
        values=values, shape=shape, step=step, bounds=(minx, miny, maxx, maxy),
        indicer=indicer
    )


@log_with(arguments=False, timed=True)
def values_to_grid(geom: BaseGeometry, grid, indicer, value: float):
    """Set values on discrete grid using scikit-image

    Args:
        geom (BaseGeometry):
            Shapely shape
        
        grid (np.ndarray):
            Grid to set values
            
        indicer (Callable): 
            Function that converts points to indices of a discrete grid. 

        value (float):
            Value to set to the grid points
    """
    for x, y in geom_to_skimage(geom, indicer):
        grid[y, x] = value


# Maps

@log_with(arguments=False, timed=True)
def distance_map(mgrid: MeshGrid,
                 targets: BaseGeometry,
                 obstacles: Optional[BaseGeometry]) -> DistanceMap:
    r"""Distance map

    Distance map :math:`S(\mathbf{x})` is obtained by solving *Eikonal equation*
    using fast marching *Fast Marching Method (FMM)* (``scikit-fmm``).

    .. math::
       \left \| \nabla S(\mathbf{x}) \right \| = \frac{1}{f(\mathbf{x})}, \quad \mathbf{x} \in \Omega

    where

    - Function :math:`f : \bar{\Omega} \mapsto (0, +\infty)` is the speed of
      travel at :math:`\mathbf{x}`

    Boundary conditions for the distance map

    .. math::
       \begin{cases}
       S(\mathbf{x}) = 0, & \mathbf{x} \in \mathcal{E} \\
       S(\mathbf{x}) \to \infty, & \mathbf{x} \in \mathcal{O}
       \end{cases}

    Initial values for speed

    .. math::
       \begin{cases}
       f(\mathbf{x}) = 1, & \mathbf{x} \in \Omega \setminus \mathcal{O} \\
       f(\mathbf{x}) \to 0, & \mathbf{x} \in \mathcal{O}
       \end{cases}

    Args:
        mgrid (MeshGrid):

        obstacles (BaseGeometry, optional):
            Impassable regions :math:`\mathcal{O}` in the domain.

        targets (BaseGeometry, optional):
            Target regions :math:`\mathcal{E}` in the domain.

    Return:
        (numpy.ndarray, numpy.ndarray, numpy.ma.MaskedArray):
            List of
            - ``mgrid``
            - ``dmap``
            - ``phi``
    """
    # Numerical values for objects in the domain
    empty_region = -1.0
    target_region = 1.0
    obstacle_region = True
    non_obstacle_region = False

    # Contour used for solving distance map
    # Mask for masked values that represent obstacles for the solver
    contour = np.full(mgrid.shape, empty_region, dtype=np.float64)
    mask = np.full(mgrid.shape, non_obstacle_region, dtype=np.bool_)

    values_to_grid(targets, contour, mgrid.indicer, target_region)
    if obstacles is not None:
        values_to_grid(obstacles, mask, mgrid.indicer, obstacle_region)

    # Solve distance map using Fast-Marching Method (FMM)
    phi = np.ma.MaskedArray(contour, mask)
    dmap = skfmm.distance(phi, dx=mgrid.step)

    return dmap


def travel_time_map(step, domain, targets, obstacles, agents):
    r"""
    Dynamics potential takes into account the positions of the agents in the
    field. Equation

    .. math::
       \left \| \nabla T(\mathbf{x}) \right \| = \frac{1}{f(\mathbf{x})}, \quad \mathbf{x} \in \Omega

    and boundary conditions

    .. math::
       f(\mathbf{x}) &= 1, \quad \mathbf{x} \in \Omega \setminus (\mathcal{O} \cup \mathcal{A}) \\
       f(\mathbf{x}) &\leq 1, \quad \mathbf{x} \in \mathcal{A} \\
       f(\mathbf{x}) &\to 0, \quad \mathbf{x} \in \mathcal{O}

    .. math::
       \frac{1}{f(\mathbf{x})} = 1 + \max \left( 0, c_{0} \left( 1 + c_{1} \frac{\mathbf{v} \cdot \nabla S(\mathbf{x})}{v_{0} \| \nabla S(\mathbf{x}) \|} \right) \right)

    - :math:`c_{0} > 0` general impact strength
    - :math:`c_{1} > 0` impact of the moving direction of an agent

    Args:
        step (float):
            Step size for the meshgrid.

        domain (Polygon):
            Domain :math:`\Omega` containing obstacles and targets.

        obstacles (LineString, optional):
            Impassable regions :math:`\mathcal{O}` in the domain.

        targets (LineString, optional):
            Target regions :math:`\mathcal{E}` in the domain.

        agents:
            Regions occupied by agents :math:`\mathcal{A}`

    Returns:
        (numpy.meshgrid, numpy.ndarray, numpy.ma.MaskedArray):
    """
    return NotImplementedError


@log_with(arguments=False, timed=True)
def direction_map(dmap: DistanceMap) -> DirectionMap:
    r"""Normalized gradient of distance map.

    Direction map is not defined when length of the gradient is zero.

    .. math::
       \hat{\mathbf{e}}_{S} = -\frac{\nabla S(\mathbf{x})}{\| \nabla S(\mathbf{x}) \|}

    Args:
        dmap (numpy.ndarray):
            Distance map.

    Returns:
        numpy.ndarray:
            Direction map. Array of shape: ``dmap.shape + (2,)``
    """
    # TODO: remove artifacts inside the obstacle regions
    u, v = np.gradient(dmap)
    l = np.hypot(u, v)
    l[l == 0] = np.nan  # Avoids zero division
    # Flip order from (row, col) to (x, y)
    return v / l, u / l


# Potentials

@log_with(timed=True)
def static_potential(domain: Polygon,
                     targets: BaseGeometry,
                     obstacles: Optional[BaseGeometry],
                     step: float,
                     radius: float,
                     value: float) -> \
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
        value (float):
        radius (float):

    Returns:
        Tuple[MeshGrid, DirectionMap]:
    """
    # Compute meshgrid for solving distance maps.
    mgrid = meshgrid(step, *domain.bounds)

    # Direction map (vector field) for guiding agents towards targets without
    # them walking into obstacles.
    obstacles_buffered = obstacles.buffer(radius).intersection(domain)
    dmap_targets = distance_map(mgrid, targets, obstacles_buffered)
    dir_map_targets = direction_map(dmap_targets)
    fill_missing(mgrid, dir_map_targets)

    # Direction map guiding agents away from the obstacles
    dmap_obs = distance_map(mgrid, obstacles, None)
    dir_map_obs = direction_map(dmap_obs)

    # Direction map that combines the two direction maps
    dir_map = merge_dir_maps(dmap_obs, dir_map_obs, dir_map_targets, radius,
                             value)

    return mgrid, dmap_targets, dir_map


def dynamic_potential():
    r"""Dynamic potential

    Navigation algorithm that takes into account the
    space that is occupied by dynamic agents (aka agents).
    """
    return NotImplementedError


algorithms = {
    'static_potential': static_potential,
    'dynamic_potential': dynamic_potential
}


# Navigator


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


def navigation(position, old_direction, mgrid, dir_map):
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
    indices = np.fliplr(mgrid.indicer(position))
    return getdefault(indices, dir_map, old_direction)

    # x, y = dir_map
    # inds = indices[:, 0], indices[:, 1]  # Has to be tuple
    # return np.stack((x[inds], y[inds]), axis=1)
