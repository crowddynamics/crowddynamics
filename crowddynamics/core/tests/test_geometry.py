from hypothesis.core import given

from crowddynamics import testing
from crowddynamics.core.geometry import geom_to_linesegment, geom_to_array


@given(geom=testing.points() |
            testing.linestrings(num_verts=3) |
            testing.polygons(num_verts=4))
def test_geom_to_array(geom):
    array = geom_to_array(geom)
    assert True


@given(geom=testing.points() |
            testing.linestrings(num_verts=3) |
            testing.polygons(num_verts=4))
def test_geom_to_linesegment(geom):
    array = geom_to_linesegment(geom)
    assert True
