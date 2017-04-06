from hypothesis import given
from hypothesis import settings
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

import crowddynamics.strategies


@given(crowddynamics.strategies.polygon(a=-1.0, b=1.0))
@settings(max_examples=500)
def test_polygon(polygon):
    minx, miny, maxx, maxy = polygon.bounds

    assert polygon.area > 0
    assert -1.0 <= minx < maxx <= 1.0
    assert -1.0 <= miny < maxy <= 1.0


@given(crowddynamics.strategies.field())
def test_field(field):
    domain, targets, obstacles = field

    assert isinstance(domain, Polygon)
    assert isinstance(targets, BaseGeometry)
    assert isinstance(obstacles, BaseGeometry)

    assert domain.area > 0

    assert targets.length > 0
    # assert targets.area > 0
    assert targets.length > 0
    # assert targets.area > 0
