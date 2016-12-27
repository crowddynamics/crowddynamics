"""
UnitTests and property based testing using
"""
import numpy as np
import pytest

from bokeh.plotting import figure, output_file, save
from hypothesis import given
from shapely.geometry import Polygon, Point

from crowddynamics.core.sampling import PolygonSample, triangle_area, \
    random_sample_triangle, triangle_area_cumsum
from tests.conf import *
from tests.strategies import polygons, vector


def save_plot(name, polygon, points):
    """

    Args:
        name (str):
        polygon (Polygon):
        points (List[Point]):

    Returns:
        None:
    """
    # TODO: name unique
    ext = ".html"
    os.makedirs(os.path.join(BASE_DIR, OUTPUT_FOLDER), exist_ok=True)
    filename = os.path.join(OUTPUT_FOLDER, name) + ext
    title = name.replace("_", "").capitalize()

    output_file(filename, title)

    # Figure
    p = figure()

    # Polygon as a patch
    values = np.asarray(polygon.exterior)
    p.patch(values[:, 0], values[:, 1], alpha=0.5, line_width=0.1)

    # Points as circles
    for point in points:
        x, y = point.xy
        p.circle(x, y)

    # TODO: save html
    save(p, filename, title=title)


@given(vector(), vector(), vector())
def test_triangle_area(a, b, c):
    # triangle = Polygon((a, b, c))
    area = triangle_area(a, b, c)
    assert isinstance(area, float)
    if np.isnan(area):
        assert True
    else:
        assert area >= 0.0


@given(vector(), vector(), vector())
def test_random_sample_triangle(a, b, c):
    triangle = Polygon((a, b, c))
    # Touches instead of contains behaves well is triangle.area is zero.
    if triangle_area(a, b, c) == 0.0:
        with pytest.raises(Exception):
            random_sample_triangle(a, b, c)
    else:
        p = random_sample_triangle(a, b, c)
        assert isinstance(p, np.ndarray)
        assert triangle.contains(Point(p))


# @given()
# def test_triangle_area_cumsum(trimesh):
#     cumsum = triangle_area_cumsum(trimesh)
#     assert True


@given(polygons)
def test_polygon_sampling(polygon):
    num = 100
    sample = PolygonSample(polygon)
    points = [Point(sample.draw()) for _ in range(num)]
    # save_plot("???", polygon, points)
    assert all(polygon.contains(p) for p in points)
