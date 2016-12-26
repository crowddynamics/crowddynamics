"""
UnitTests and property based testing using
"""
import os
import unittest

from bokeh.plotting import figure, output_file, save
from shapely.geometry import Polygon, Point

from crowddynamics.core.sampling import PolygonSample
from crowddynamics.core.vector2D import *
from tests.strategies import generate_polygon

OUTPUT_FOLDER = "output"


def save_plot(name, polygon, points):
    """

    Args:
        name (str):
        polygon (Polygon):
        points (List[Point]):

    Returns:
        None:
    """
    ext = ".html"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
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


# -----------------------------------------------------------------------------
# Unittests
# -----------------------------------------------------------------------------


class PolygonSampleTest(unittest.TestCase):
    """
    Test case for PolygonSample.
    """

    def test_simple(self, num=100):
        polygon = generate_polygon()
        sample = PolygonSample(polygon)
        points = []

        for i in range(num):
            p = Point(sample.draw())
            points.append(p)
            self.assertTrue(polygon.contains(p), "Point not in polygon")

        save_plot("polygon_sample_simple", polygon, points)

    def test_complex(self, num=100):
        polygon = generate_polygon().buffer(0.1)
        sample = PolygonSample(polygon)
        points = []

        for i in range(num):
            p = Point(sample.draw())
            points.append(p)
            self.assertTrue(polygon.contains(p), "Point not in polygon")

        save_plot("polygon_sample_complex", polygon, points)
