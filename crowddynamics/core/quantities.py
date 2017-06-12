import numba
import numpy as np
from numba import f8, i8
from scipy.spatial import Voronoi
from shapely.geometry.polygon import Polygon

from crowddynamics.core.geom2D import polygon_area


def density_classical(points, cell_size):
    r"""Classical definition of density defined the density :math:`D` as number
    of agents per unit of area. [Steffen2010]_

    .. math::
       D = \frac{N}{|A|}\quad \left[\mathrm{\frac{1}{m^{2}}}\right]

    where

    - :math:`N` is the number of agents
    - :math:`|A|` is the size of the area

    However the classical definition of density does not work as well when
    measuring density of crowd or granular media as it does for measuring
    density of fluids which have :math:`> 10^{18}` particles per
    :math:`\mathrm{mm}^3`.
    """
    indices = np.floor(points / cell_size)
    x, y = indices[:, 0], indices[:, 1]
    x_min, y_min = x.min(), y.min()
    shape = x.max() - x_min, y.max() - y_min
    count = np.zeros(shape, dtype=np.int64)
    for i, j in zip(x, y):
        count[i - x_min, j - y_min] += 1
    return count / cell_size ** 2


@numba.jit()
def bounding_box(points):
    """Bounding box

    Args:
        points: Array of shape (amount_of_points, dimensions)

    Returns:
        numpy.ndarray: Array of [[min, max], [min, max], ...] along the
            dimensions of points.
    """
    out = np.empty((points.ndim, 2))
    for i in range(points.ndim):
        x = points[:, i]
        out[i, 0] = x.min()
        out[i, 1] = x.max()
    return out


def rectangle(xmin, xmax, ymin, ymax):
    return Polygon(((xmin, ymin),
                    (xmin, ymax),
                    (xmax, ymax),
                    (xmax, ymin)))


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions. [colorize_voronoi]

    .. [colorize_voronoi] https://gist.github.com/pv/8036995

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def density_voronoi_1(points, cell_size):
    r"""Density definition base on Voronoi-diagram. [Steffen2010]_

    Compute Voronoi-diagram for each position :math:`\mathbf{p}_i` giving
    cells :math:`A_i` to each agent :math:`i`. Area of Voronoi cell is
    denoted :math:`|A_i|`. Area :math:`A` is the area inside which we
    measure the density.
    For set of agents :math:`S = \{i : \mathbf{q} \in A_i \wedge \mathbf{q} \in A\}`
    whose Voronoi-diagram has points that belong to area :math:`A`.

    Density distribution

    .. math::
        p_i(\mathbf{x}) &=
        \frac{1}{|A_i|}
        \begin{cases}
            1 & \mathbf{x} \in A_i \\
            0 & \text{otherwise}
        \end{cases} \\
        p(\mathbf{x}) &= \sum_{i \in S} p_i(\mathbf{x})

    Density

    .. math::
        D_V &= \frac{1}{|A|} \int_A p(\mathbf{x}) d\mathbf{x} \\
            &= \frac{1}{|A|} \int_A \sum_{i \in S} p_i(\mathbf{x}) d\mathbf{x} \\
            &= \frac{1}{|A|} \sum_{i \in S} \int_A  p_i(\mathbf{x}) d\mathbf{x} \\
            &= \frac{1}{|A|} \sum_{i \in S} \frac{1}{|A_i|} \int_A
               \begin{cases}
                   1 & \mathbf{x} \in A_i \\
                   0 & \text{otherwise}
               \end{cases}
               d\mathbf{x} \\
            &= \frac{1}{|A|} \sum_{i \in S} \frac{|A \cap A_i|}{|A_i|}

    Args:
        points (numpy.ndarray):
            Two dimensional points :math:`\mathbf{p}_{i \in \mathbb{N}}`
        cell_size (float):
            Cell size of the meshgrid. Each cell represents an area :math:`A`
            inside which density is measured.

    Returns:
        numpy.ndarray: Grid of values density values for each cell.

    """
    assert points.ndim == 2, 'Points should be two dimensional.'
    assert len(points) >= 3, 'Three of more points should be supplied.'
    assert cell_size > 0, 'Cell size should be positive number (cell_size > 0).'

    # TODO: add obstacles
    # TODO: observation area should be settable by hand
    # TODO: filter out points that do not belong to observation area
    # TODO: optimized version for small cell_size using scikit-image drawing
    # polygon and polygon perimiters separately

    bbox_points = bounding_box(points)
    (xmin, xmax), (ymin, ymax) = bbox_points
    observation_area = rectangle(xmin, xmax, ymin, ymax)

    # Voronoi tesselation
    vor = Voronoi(points)
    new_regions, new_vertices = voronoi_finite_polygons_2d(vor)

    # Density matrix
    (imin, imax), (jmin, jmax) = np.int64(bbox_points / cell_size)
    density = np.zeros(shape=(imax - imin + 1, jmax - imin + 1))

    # Loop over Voronoi regions
    for region in new_regions:
        vertices = new_vertices[region]
        voronoi_cell = Polygon(shell=vertices) & observation_area
        minx, miny, maxx, maxy = voronoi_cell.bounds

        # Loop over the cells contained by the bounding box of the Voronoi cell
        # FIXME: index out of bounds
        for i in range(int(minx / cell_size), int(maxx / cell_size) + 1):
            for j in range(int(miny / cell_size), int(maxy / cell_size) + 1):
                cell = rectangle(i * cell_size, (i + 1) * cell_size,
                                 j * cell_size, (j + 1) * cell_size)
                intersection = voronoi_cell & cell
                density[i - imin, j - jmin] += intersection.area / voronoi_cell.area

    return density / (cell_size ** 2)


def density_voronoi_2(points, cell_size):
    r"""Density definition base on Voronoi-diagram. [Steffen2010]_

    Compute Voronoi-diagram for each position :math:`\mathbf{p}_i` giving
    cells :math:`A_i` to each agent :math:`i`. Area of Voronoi cell is
    denoted :math:`|A_i|`. Area :math:`A` is the area inside which we
    measure the density.

    :math:`S = \{i : \mathbf{p}_i \in A\}` is the set of agents and
    :math:`N = |S|` the number of agents that is inside area :math:`A`.

    .. math::
        D_{V^\prime} = \frac{N}{\sum_{i \in S} |A_i|}

    Args:
        points (numpy.ndarray):
            Two dimensional points :math:`\mathbf{p}_{i \in \mathbb{N}}`
        cell_size (float):
            Cell size of the meshgrid. Each cell represents an area :math:`A`
            inside which density is measured.

    Returns:
        numpy.ndarray: Grid of values density values for each cell.

    """
    # Voronoi tesselation
    vor = Voronoi(points)
    new_regions, new_vertices = voronoi_finite_polygons_2d(vor)

    # Compute areas for all of the Voronoi regions
    area = np.array([polygon_area(new_vertices[i]) for i in new_regions])

    return _core_2(points, cell_size, area, vor.point_region)


@numba.jit([f8[:, :](f8[:, :], f8, f8[:], i8[:])],
           nopython=True, nogil=True, cache=True)
def _core_2(points, cell_size, area, point_region):
    """Numerical implementation

    Args:
        points:
        cell_size:
        area:
        point_region:

    Returns:

    """
    # FIXME
    indices = (points / cell_size).astype(np.int64)
    x, y = indices[:, 0], indices[:, 1]
    x_min, y_min = x.min(), y.min()
    shape = x.max() - x_min + 1, y.max() - y_min + 1
    density = np.zeros(shape)
    area_sum = np.zeros(shape)

    for k, (i, j) in enumerate(zip(x, y)):
        index = i - x_min, j - y_min
        density[index] += 1
        area_sum[index] += area[point_region[k]]

    n, m = density.shape
    for i in range(n):
        for j in range(m):
            s = area_sum[i, j]
            if s > 0:
                density[i, j] /= s

    return density
