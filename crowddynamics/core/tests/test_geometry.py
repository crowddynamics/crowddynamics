from shapely.geometry import Polygon, LineString, Point

from crowddynamics.core.geometry import geom_to_linesegment, geom_to_mpl

pts = [(0.0, 0.0), (1.0, 1.0), (0.0, 2.0)]

point = Point((0.0, 0.0))
lstring = LineString(pts)
polygon = Polygon(pts)
multi = point | lstring | polygon


def _segments(p):
    return zip(p[:-1], p[1:])


def test_geom_to_linesegment():
    assert list(geom_to_linesegment(point)) == []
    assert list(geom_to_linesegment(lstring)) == \
           list(_segments(pts))
    assert list(geom_to_linesegment(polygon)) == \
           list(_segments(pts + [pts[0]]))
    geom_to_linesegment(multi)
    assert True


def test_geom_to_mpl():
    # geom_to_mpl(point)
    geom_to_mpl(lstring)
    geom_to_mpl(polygon)
    geom_to_mpl(multi)
    assert True
