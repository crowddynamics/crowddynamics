"""
Sampling
"""
import numba
import numpy as np
from numba import f8
from scipy.spatial.qhull import Delaunay
from shapely.geometry import Polygon


@numba.jit(f8(f8[:], f8[:], f8[:]), nopython=True, nogil=True)
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


@numba.jit(nopython=True, nogil=True)
def triangle_area_cumsum(trimesh):
    r"""
    Computes cumulative sum of the areas of the triangle mesh.

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


@numba.jit(f8[:](f8[:], f8[:], f8[:]), nopython=True, nogil=True)
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


class PolygonSample:
    r"""
    Uniform sampling of convex polygon

    Generates random uniform point from inside of polygon.

    1) `Delaunay triangulation`_ to break the polygon into triangular mesh.
    2) Draw random uniform triangle weighted by its area.
    3) Draw random uniform sample from inside the triangle.

    .. _Delaunay triangulation: https://en.wikipedia.org/wiki/Delaunay_triangulation

    References:

        http://gis.stackexchange.com/questions/6412/generate-points-that-lie-inside-polygon
    """
    def __init__(self, polygon):
        r"""
        Convex Polygon to sample. (Non convex polygon will be treated as convex)

        Args:
            polygon (Polygon | numpy.ndarray): Shapely polygon or numpy array of
                polygon vertices.

        """
        if isinstance(polygon, Polygon):
            self.vertices = np.asarray(polygon.exterior)
        else:
            # Should be numpy.ndarray
            self.vertices = polygon

        # Triangular mesh using Delaunay triangulation algorithm
        # FIXME: Delaunay only works for convex polygon
        self.delaunay = Delaunay(self.vertices)
        self.mesh = self.vertices[self.delaunay.simplices]

        # Cumulative sum of areas of the triangles
        self.weights = triangle_area_cumsum(self.mesh)
        self.weights /= self.weights[-1]  # Normalize values to interval [0, 1]

    def draw(self):
        r"""
        Draw random triangle weighted by the area of the triangle and draw
        random sample

        Returns:
            numpy.ndarray: Uniformly sampled point inside the polygon

        """
        x = np.random.random()
        i = np.searchsorted(self.weights, x)
        a, b, c = self.mesh[i]
        sample = random_sample_triangle(a, b, c)
        return sample

    def generator(self, num=None):
        r"""
        Generator that generates ``num`` amount of draws.

        Args:
            num (int, optional):
                Number of draws, if None returns infinite generator.

        Returns:
            Generator:
        """
        i = 0
        while num is None or i < num:
            yield self.draw()
            i += 1


class LineStringSample:
    # TODO: LineStringSampling
    pass
