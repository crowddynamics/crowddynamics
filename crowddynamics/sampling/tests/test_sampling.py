"""
UnitTests and property based testing using
"""
import os
import sys

import numpy as np
import pytest
from bokeh.plotting import figure, output_file, save
from hypothesis import given, note, assume
from shapely.geometry import Polygon, Point

from crowddynamics.sampling import PolygonSample, triangle_area, \
    random_sample_triangle, triangle_area_cumsum
from crowddynamics.tests.strategies import polygons, vector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = "output"

EPSILON = sys.float_info.epsilon


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
    area = triangle_area(a, b, c)
    assert isinstance(area, float)
    assert np.isnan(area) or area >= 0.0


@given(vector(), vector(), vector())
def test_random_sample_triangle(a, b, c):
    # Assume that the area of the triangle is not zero.
    area = triangle_area(a, b, c)
    assume(not np.isclose(area, 0.0))

    triangle = Polygon((a, b, c))
    p = random_sample_triangle(a, b, c)
    point = Point(p)
    distance = triangle.distance(point)
    note(
        r"""
        area: {}
        a: {}
        b: {}
        c: {}
        p: {}
        distance: {}
        """.format(area, a, b, c, p, distance)
    )
    assert isinstance(p, np.ndarray)
    assert triangle.intersects(point) or np.isclose(distance, 0.0)


@pytest.mark.skip
def test_triangle_area_cumsum(trimesh):
    cumsum = triangle_area_cumsum(trimesh)
    assert True


@pytest.mark.skip
@given(polygons)
def test_polygon_sampling(polygon):
    num = 100
    sample = PolygonSample(polygon)
    points = [Point(sample.draw()) for _ in range(num)]
    # save_plot("???", polygon, points)
    assert all(polygon.contains(p) for p in points)
