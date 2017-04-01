import numba
from numba import f8
import numpy as np
from scipy.spatial.qhull import Delaunay
from shapely.geometry import Polygon, Point


@numba.jit(f8(f8[:], f8[:], f8[:]), nopython=True, nogil=True)
def triangle_area(a, b, c):
    return np.abs(a[0] * (b[1] - c[1]) +
                  b[0] * (c[1] - a[1]) +
                  c[0] * (a[1] - b[1])) / 2


@numba.jit(f8[:](f8[:], f8[:], f8[:]), nopython=True, nogil=True)
def random_sample_triangle(a, b, c):
    """
    Uniform sampling of a triangle
    ------------------------------
    Generate uniform random sample from a triangle defined by points A, B
    and C [1]_, [2]_. Point inside the triangle is given

    .. math::
       P = (1 - \sqrt{r_1}) A + (\sqrt{r_1} (1 - r_2))  B + (r_2 \sqrt{r_1}) C,

    where random variables are

    .. math::
       r_1, r_2 \sim \mathcal{U}(0, 1)

    References
    ----------
    .. [1] http://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle
    .. [2] http://mathworld.wolfram.com/TrianglePointPicking.html

    :param a: Vertex of the triangle
    :param b: Vertex of the triangle
    :param c: Vertex of the triangle
    :return: Uniformly distributed random point P.
    """
    # Random variables
    r1 = np.random.random()
    r2 = np.random.random()
    return (1 - np.sqrt(r1)) * a + \
           np.sqrt(r1) * (1 - r2) * b + \
           r2 * np.sqrt(r1) * c


@numba.jit(nopython=True, nogil=True)
def triangle_area_cumsum(mesh):
    """Computes cumulative sum of the areas of the triangle mesh.

    :param mesh: Trianle mesh.
    :return: Cumulative sum the area of the triangle mesh
    """
    area = 0
    rows = mesh.shape[0]
    cumsum = np.zeros(rows)
    for i in range(rows):
        a, b, c = mesh[i, 0, :], mesh[i, 1, :], mesh[i, 2, :]
        area += triangle_area(a, b, c)
        cumsum[i] = area
    return cumsum


class PolygonSample:
    """
    Uniform sampling of convex polygon
    ----------------------------------
    Generates random uniform point from inside of polygon. [1]_

    - Delaunay triangulation to break the polygon into triangular mesh. [2]_
    - Draw random uniform triangle weighted by its area.
    - Draw random uniform sample from inside the triangle.

    .. [1] http://gis.stackexchange.com/questions/6412/generate-points-that-lie-inside-polygon
    .. [2] https://en.wikipedia.org/wiki/Delaunay_triangulation
    """

    def __init__(self, polygon):
        """Polygon to sample.

        :param polygon: Shapely polygon or numpy array of polygon vertices.
        """
        if isinstance(polygon, Polygon):
            self.vertices = np.asarray(polygon.exterior)
        elif isinstance(polygon, np.ndarray):
            self.vertices = polygon
        else:
            raise Exception("")

        # Triangular mesh by Delaunay triangulation algorithm
        self.delaunay = Delaunay(self.vertices)
        self.mesh = self.vertices[self.delaunay.simplices]

        # Cumulative sum of areas of the triangles
        self.weights = triangle_area_cumsum(self.mesh)
        self.weights /= self.weights[-1]  # Normalize values to interval [0, 1]

    def draw(self):
        # Draw random triangle weighted by the area of the triangle and draw
        # random sample
        x = np.random.random()
        i = np.searchsorted(self.weights, x)
        a, b, c = self.mesh[i]
        sample = random_sample_triangle(a, b, c)
        return Point(sample)
