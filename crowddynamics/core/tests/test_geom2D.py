import numpy as np
from hypothesis import given, assume
from shapely.geometry import Polygon

from crowddynamics.core.geom2D import polygon_area
from crowddynamics.testing import real


@given(real(-10.0, 10.0, shape=(5, 2)))
def test_polygon_area(vertices):
    poly = Polygon(vertices).convex_hull
    assume(poly.area > 0.0)
    area = polygon_area(np.asarray(poly.exterior))
    assert np.isclose(area, poly.area)
