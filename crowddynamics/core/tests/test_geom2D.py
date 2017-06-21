import numpy as np
import pytest
from hypothesis import given, assume
from shapely.geometry import Polygon, LineString

from crowddynamics.core.geom2D import polygon_area, line_intersect
from crowddynamics.core.vector2D import length
from crowddynamics.testing import reals


@given(reals(-10.0, 10.0, shape=(5, 2)))
def test_polygon_area(vertices):
    poly = Polygon(vertices).convex_hull
    assume(poly.area > 0.0)
    area = polygon_area(np.asarray(poly.exterior))
    assert np.isclose(area, poly.area)


# @pytest.mark.skip('Fix line_intersect function')
@given(x0=reals(0.0, 1.0, shape=2, exclude_zero='near'),
       x1=reals(0.0, 1.0, shape=2, exclude_zero='near'),
       y0=reals(0.0, 1.0, shape=2, exclude_zero='near'),
       y1=reals(0.0, 1.0, shape=2, exclude_zero='near'))
def test_line_intersect(x0, x1, y0, y1):
    assume(not np.isclose(length(x1 - x0), 0.0))
    assume(not np.isclose(length(y1 - y0), 0.0))

    res = line_intersect(x0, x1, y0, y1)
    correct = LineString([x0, x1]).intersects(LineString([y0, y1]))
    assert res == correct
