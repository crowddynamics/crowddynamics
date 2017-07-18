"""UnitTests and property based testing"""
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, assume
from hypothesis.extra.numpy import arrays
from shapely.geometry import Polygon, Point

from crowddynamics.core.geom2D import polygon_area
from crowddynamics.core.sampling import random_sample_triangle, \
    triangle_area_cumsum, polygon_sample, \
    linestring_sample
from crowddynamics.testing import reals

# Convex shapes
triangle = Polygon([(1.0, 1.0), (0.0, 0.0), (2.0, 0.0)])
square = Polygon([(1.0, 1.0), (0.0, 0.0), (2.0, 0.0), (3.0, 0.5)])
tetragon = Polygon([(1.0, 1.0), (0.0, 0.0), (2.0, 0.0), (3.0, 0.5), (2.5, 2.0)])


@pytest.mark.parametrize('poly', (triangle, square, tetragon))
def test_linestring_sample(poly):
    ls = poly.exterior
    vertices = np.asarray(ls)
    for i, p in zip(range(10), linestring_sample(vertices)):
        assert np.isclose(ls.distance(Point(p)), 0.0)


@given(arrays(np.float64, (10, 3, 2), st.floats(-100, 100, False, False)))
def test_triangle_area_cumsum(trimesh):
    cumsum = triangle_area_cumsum(trimesh)
    assert isinstance(cumsum, np.ndarray)
    assert cumsum.dtype.type is np.float64
    assert np.all(np.sort(cumsum) == cumsum)


@given(reals(min_value=-100, max_value=100, shape=2),
       reals(min_value=-100, max_value=100, shape=2),
       reals(min_value=-100, max_value=100, shape=2))
def test_random_sample_triangle(a, b, c):
    # Assume that the area of the triangle is not zero.
    area = polygon_area(np.stack((a, b, c)))
    assume(not np.isclose(area, 0.0))

    triangle = Polygon((a, b, c))
    p = random_sample_triangle(a, b, c)
    point = Point(p)
    distance = triangle.distance(point)

    assert isinstance(p, np.ndarray)
    assert triangle.intersects(point) or np.isclose(distance, 0.0)


@pytest.mark.parametrize('poly', (triangle, square, tetragon))
def test_polygon_sampling(poly):
    # Numerical error is too great if area of the polygon is too small
    poly = poly.convex_hull
    # assume(poly.area > 0.01)
    exterior = np.asarray(poly.exterior)

    for i, point in zip(range(100), polygon_sample(exterior)):
        assert poly.contains(Point(point))
