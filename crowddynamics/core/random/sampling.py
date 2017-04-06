"""Sampling points on polygons."""
import numba
import numpy as np
from numba import f8
from scipy.spatial.qhull import Delaunay


@numba.jit(f8(f8[:], f8[:], f8[:]), nopython=True, nogil=True, cache=True)
def triangle_area(a, b, c):
    r"""
    Area of a triangle given by points :math:`\mathbf{a}`, :math:`\mathbf{b}`,
    and :math:`\mathbf{c}`.

    .. math::
       \frac{|a_0 (b_1 - c_1) + b_0 (c_1 - a_1) + c_0 (a_1 - b_1)|}{2}

    Args:
        a (numpy.ndarray): Vertex of the triangle
        b (numpy.ndarray): Vertex of the triangle
        c (numpy.ndarray): Vertex of the triangle

    Returns:
        float: Area of the triangle

    """
    return np.abs(a[0] * (b[1] - c[1]) +
                  b[0] * (c[1] - a[1]) +
                  c[0] * (a[1] - b[1])) / 2


@numba.jit([f8[:](f8[:, :, :])], nopython=True, nogil=True, cache=True)
def triangle_area_cumsum(trimesh):
    r"""Computes cumulative sum of the areas of the triangle mesh.

    Args:
        trimesh (numpy.ndarray):
            Triangle mesh array of shape=(n, 3, 2)

    Returns:
        numpy.ndarray:
            Cumulative sum the area of the triangle mesh
    """
    area = 0.0
    rows = trimesh.shape[0]
    cumsum = np.zeros(rows)
    for i in range(rows):
        a, b, c = trimesh[i, 0, :], trimesh[i, 1, :], trimesh[i, 2, :]
        area += triangle_area(a, b, c)
        cumsum[i] = area
    return cumsum


@numba.jit(f8[:](f8[:], f8[:], f8[:]), nopython=True, nogil=True, cache=True)
def random_sample_triangle(a, b, c):
    r"""
    Generate uniform random sample from inside of a triangle defined by points
    :math:`a`, :math:`b` and :math:`c`. [1]_

    .. math::
       P = (1 - \sqrt{R_1}) \mathbf{a} + (\sqrt{R_1} (1 - R_2)) \mathbf{b} +
       (R_2 \sqrt{R_1}) \mathbf{c},

    where uniformly distributed random variables are

    .. math::
       R_1, R_2 \sim \mathcal{U}(0, 1)

    Does not work for triangles that have area of zero.

    Args:
        a (numpy.ndarray): Vertex of the triangle
        b (numpy.ndarray): Vertex of the triangle
        c (numpy.ndarray): Vertex of the triangle

    Returns:
        numpy.ndarray: Uniformly distributed random point P

    References:
        .. [1] http://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle
    """
    # assume triangle.area is not close to 0.0
    r1 = np.random.random()
    r2 = np.random.random()
    return (1 - np.sqrt(r1)) * a + \
           np.sqrt(r1) * (1 - r2) * b + \
           r2 * np.sqrt(r1) * c


def linestring_sample():
    pass


def polygon_sample(vertices):
    """Uniform sampling of points inside a convex polygon. Non convex polygons 
    will be treated as the input would be their convex hull.
    
    Steps of the algorithm
    
    1) `Delaunay triangulation`_ to break the polygon into triangular mesh.
    2) Draw random uniform triangle weighted by its area.
    3) Draw random uniform sample from inside the triangle.
    
    .. _Delaunay triangulation: https://en.wikipedia.org/wiki/Delaunay_triangulation
    
    Args:
        vertices (numpy.ndarray): 
            Array of polygon vertices. Shape of (n, 2).

    Yields:
        numpy.ndarray: 
            Random point inside the polygon.

    References:
        - http://gis.stackexchange.com/questions/6412/generate-points-that-lie-inside-polygon
    
    Todo:
        - Algorithm for sampling non-convex polygons
    """
    assert len(vertices.shape) == 2
    assert vertices.shape[1] == 2

    delaunay = Delaunay(vertices)  # Triangulation
    mesh = vertices[delaunay.simplices]  # Triangle mesh

    # Weights for choosing random uniform triangle from the mesh.
    # Weight are normalized to values in interval [0, 1].
    weights = triangle_area_cumsum(mesh)
    weights /= weights[-1]

    while True:
        x = np.random.random()  # Random variable from interval [0, 1]
        i = np.searchsorted(weights, x)  # Uniformly drawn random triangle
        a, b, c = mesh[i]
        yield random_sample_triangle(a, b, c)
