from shapely.geometry import Polygon, LineString, Point

from crowddynamics.core.geometry import geom_to_linesegment


def _segments(p): return zip(p[:-1], p[1:])


def test_geom_to_linesegment():
    pts = [(0.0, 0.0), (1.0, 1.0), (0.0, 2.0)]

    assert list(geom_to_linesegment(Point((0.0, 0.0)))) == []
    assert list(geom_to_linesegment(LineString(pts))) == \
           list(_segments(pts))
    assert list(geom_to_linesegment(Polygon(pts))) == \
           list(_segments(pts + [pts[0]]))
