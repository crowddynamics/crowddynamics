"""
UnitTests and property based testing using
"""
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, assume
from hypothesis.extra.numpy import arrays
from shapely.geometry import Polygon, Point

from crowddynamics.core.random.sampling import PolygonSample, triangle_area, \
    random_sample_triangle, triangle_area_cumsum
from crowddynamics.testing import vector, real, polygon


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


@given(polygon(a=-1.0, b=1.0, num_points=5, convex_hull=True))
def test_polygon_sampling(polygon):
    # Numerical error is too great if area of the polygon is too small
    assume(polygon.area > 0.01)

    sample_size = 20
    sample = PolygonSample(polygon)
    for point in sample.generator(sample_size):
        assert polygon.contains(Point(point))
