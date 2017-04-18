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
        
Todo: 
    - improve performance
"""
from collections import namedtuple
from typing import Tuple, Optional

import numba
import numpy as np
import skfmm
from numba import f8, i8
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from crowddynamics.core.geometry import geom_to_skimage

MeshGrid = namedtuple('MeshGrid', 'values shape step bounds indicer')
DistanceMap = np.ma.MaskedArray
DirectionMap = Tuple[np.ma.MaskedArray, np.ma.MaskedArray]


# Numerical Routines

def interpolate_direction_map(mgrid: MeshGrid,
                              dir_map: DirectionMap) -> DirectionMap:
    """Interpolate direction map

    Args:
        mgrid:
        dir_map:

    Returns:
        DirectionMap:
    """
    from scipy.interpolate import griddata

    x, y = mgrid.values
    u, v = dir_map

    # Invert boolean values, because we want non masked values
    nomask = u.mask ^ True

    # Stack into shape (n, 2)
    points = np.stack((y[nomask], x[nomask])).T

    u_out = griddata(points, u[nomask], xi=(y, x), method='nearest')
    v_out = griddata(points, v[nomask], xi=(y, x), method='nearest')

    return u_out, v_out


@numba.jit((f8[:, :], numba.types.Tuple((f8[:, :], f8[:, :])),
            numba.types.Tuple((f8[:, :], f8[:, :])), f8, f8),
           nopython=True, nogil=True, cache=True)
def merge_dir_maps(dmap, dir_map1, dir_map2, radius, strength):
    r"""
    Function that merges two direction maps together. Let distance map from
    obstacles be :math:`\Phi(\mathbf{x})` and :math:`\lambda(\Phi(\mathbf{x}))`
    be any decreasing function :math:`\lambda^{\prime}(\Phi(\mathbf{x})) < 0` of
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
        dir_map1:
        dir_map2:
        radius (float):
            Radius
        strength (float):
            Value between (0, 1). Value denoting the strength of dir_map1 at
            distance of radius.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
    """
    # FIXME: artifacts near radius distance from obstacles
    u1, v1 = dir_map1  # Obstacles
    u2, v2 = dir_map2  # Targets
    u_out, v_out = np.copy(u2), np.copy(v2)

    n, m = dmap.shape
    eps = -4.0e-08
    for i in range(n):
        for j in range(m):
            # Distance from the obstacles
            x = np.abs(dmap[i, j])
            if x - radius < eps:
                # Decreasing function
                k = strength ** (x / radius)
                u_out[i, j] = - k * u1[i, j] + (1 - k) * u2[i, j]
                v_out[i, j] = - k * v1[i, j] + (1 - k) * v2[i, j]

    l = np.hypot(u_out, v_out)
    return u_out / l, v_out / l


# Grid

def meshgrid(step: float, minx: float, miny: float,
             maxx: float, maxy: float) -> MeshGrid:
    """2-Dimensional meshgrid with inclusive end points maxx and maxy"""
    x = np.arange(minx, maxx + step, step=step)
    y = np.arange(miny, maxy + step, step=step)
    values = np.meshgrid(x, y, indexing='xy')
    shape = values[0].shape

    def indicer(position):
        shifted = np.asarray(position) - np.array((minx, miny))
        return (shifted // step).astype(np.int64)

    return MeshGrid(
        values=values, shape=shape, step=step, bounds=(minx, miny, maxx, maxy),
        indicer=indicer
    )


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
    for y, x in geom_to_skimage(geom, indicer):
        grid[x, y] = value


# Maps

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
        mgrid:

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

def static_potential(domain: Polygon, targets: BaseGeometry,
                     obstacles: BaseGeometry, step: float, radius: float,
                     value: float) -> Tuple[MeshGrid, DistanceMap, DirectionMap]:
    r"""Static potential

    Navigation algorithm that does not take into account
    the space that is occupied by dynamic agents (aka agents).

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
    dmap_exits = distance_map(mgrid, targets, obstacles_buffered)
    # FIXME: interpolation
    dir_map_exits = interpolate_direction_map(mgrid, direction_map(dmap_exits))

    # Direction map guiding agents away from the obstacles
    dmap_obs = distance_map(mgrid, obstacles, None)
    dir_map_obs = direction_map(dmap_obs)

    # Direction map that combines the two direction maps
    dir_map = merge_dir_maps(dmap_obs, dir_map_obs, dir_map_exits, radius,
                             value)

    return mgrid, dmap_exits, dir_map


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
def inside(a, lower, upper):
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
        if inside((i, j), (0, 0), x.shape):
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
