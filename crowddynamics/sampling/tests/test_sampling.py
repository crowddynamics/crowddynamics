"""
UnitTests and property based testing using
"""
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, assume
from hypothesis import note
from hypothesis.extra.numpy import arrays
from shapely.geometry import LineString
from shapely.geometry import Polygon, Point

from crowddynamics.sampling import PolygonSample, triangle_area, \
    random_sample_triangle, triangle_area_cumsum
from crowddynamics.testing import vector, real, vectors


@st.composite
def polygons(draw, min_value=-1.0, max_value=1.0, num_points=5):
    """
    Generate a random polygon. Polygon should have area > 0.

    Args:
        draw:
        min_value (float):
        max_value (float):
        num_points (int):

    Returns:
        Polygon: Random convex polygon

    """
    points = draw(arrays(np.float64, (num_points, 2),
                         real(min_value, max_value, exclude_zero='near')))
    buffer = draw(real(0.1, 0.2))
    # FIXME: Remove convex hull when sampling support None convex polygons
    return LineString(points).buffer(buffer).convex_hull


@given(vector(), vector(), vector())
def test_triangle_area(a, b, c):
    area = triangle_area(a, b, c)
    assert isinstance(area, float)
    assert np.isnan(area) or area >= 0.0


@given(arrays(np.float64, (10, 3, 2), st.floats(-100, 100, False, False)))
def test_triangle_area_cumsum(trimesh):
    cumsum = triangle_area_cumsum(trimesh)
    assert isinstance(cumsum, np.ndarray)
    assert cumsum.dtype.type is np.float64
    assert np.all(np.sort(cumsum) == cumsum)


@given(
    vector(elements=real(min_value=-100, max_value=100)),
    vector(elements=real(min_value=-100, max_value=100)),
    vector(elements=real(min_value=-100, max_value=100)),
)
def test_random_sample_triangle(a, b, c):
    # Assume that the area of the triangle is not zero.
    area = triangle_area(a, b, c)
    assume(not np.isclose(area, 0.0))

    triangle = Polygon((a, b, c))
    p = random_sample_triangle(a, b, c)
    point = Point(p)
    distance = triangle.distance(point)

    assert isinstance(p, np.ndarray)
    assert triangle.intersects(point) or np.isclose(distance, 0.0)


@given(polygons())
def test_polygon_sampling(polygon):
    assume(not np.isclose(polygon.area, 0.0))

    sample_size = 10
    sample = PolygonSample(polygon)
    for point in sample.generator(sample_size):
        assert polygon.contains(Point(point))
