"""
UnitTests and property based testing using
"""
import pytest
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, assume
from hypothesis.extra.numpy import arrays
from shapely.geometry import Polygon, Point
import sys

from crowddynamics.core.random.sampling import triangle_area, \
    random_sample_triangle, triangle_area_cumsum, polygon_sample
from crowddynamics.testing import real, polygon


@pytest.mark.skip
def test_linestring_sample():
    pass


@given(real(shape=2), real(shape=2), real(shape=2))
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


@given(real(min_value=-100, max_value=100, shape=2),
       real(min_value=-100, max_value=100, shape=2),
       real(min_value=-100, max_value=100, shape=2))
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


@given(polygon(a=-1.0, b=1.0, num_points=5))
def test_polygon_sampling(poly):
    # Numerical error is too great if area of the polygon is too small
    poly = poly.convex_hull
    assume(poly.area > 0.01)

    for i, point in zip(range(20), polygon_sample(np.asarray(poly.exterior))):
        assert poly.contains(Point(point))
