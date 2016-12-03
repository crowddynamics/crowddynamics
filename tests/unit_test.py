"""
UnitTests and property based testing using

- Unittests
- `Hypothesis <https://hypothesis.readthedocs.io/en/latest/index.html>`_

"""
import importlib
import os
import sys
import unittest

import numpy as np
from bokeh.plotting import figure, output_file, save

sys.path.insert(0, "..")
from shapely.geometry import Polygon, Point
from crowddynamics.functions import load_config
from crowddynamics.core.sampling import PolygonSample
from crowddynamics.core.vector2D import *

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


def generate_polygon(low=0.0, high=1.0, max_verts=5):
    """
    Generate a random polygon.

    Args:
        low:
        high:
        max_verts:

    Returns:
        Polygon: Random convex polygon
    """
    points = np.random.uniform(low=low, high=high, size=(max_verts, 2))
    polygon = Polygon(points).convex_hull
    return polygon


def import_simulations(queue=None):
    """
    Yield all simulations from examples.py

    Args:
        queue (multiprocessing.Queue, optional):

    Yields:
        MultiAgentSimulation:
    """
    configs = load_config("simulations.yaml")
    conf = configs["simulations"]
    for name in conf.keys():
        d = conf[name]
        module = importlib.import_module(d["module"])
        simulation = getattr(module, d["class"])
        process = simulation(queue, **d["kwargs"])
        yield process


# -----------------------------------------------------------------------------
# Unittests
# -----------------------------------------------------------------------------


class Vector2DTest(unittest.TestCase):
    def test_wrap_to_pi(self):
        self.assertEqual(wrap_to_pi(0.0), 0.0)
        self.assertEqual(wrap_to_pi(np.pi), np.pi)
        self.assertEqual(wrap_to_pi(-np.pi), -np.pi)
        self.assertEqual(wrap_to_pi(2 * np.pi), 0)
        self.assertEqual(wrap_to_pi(-2 * np.pi), 0)

    def test_rotate90(self):
        self.assertTrue(np.all(rotate90(np.array([0.0, 0.0])) == np.array([0.0, 0.0])))
        self.assertTrue(np.all(rotate90(np.array([1.0, 1.0])) == np.array([-1.0, 1.0])))

    def test_rotate270(self):
        self.assertEqual(rotate270(np.array([0.0, 0.0])), np.array([0.0, 0.0]))
        self.assertEqual(rotate270(np.array([1.0, 1.0])), np.array([1.0, -1.0]))

    def test_angle(self):
        self.assertEqual(angle(np.array([0.0, 0.0])), 0.0)
        x = np.array([-1, +1, +1, -1])
        y = np.array([-1, -1, +1, +1])
        ref = np.array([-135., -45., 45., 135.])
        for x1, y1, deg in zip(x, y, ref):
            vec = np.array([x1, y1])
            self.assertEqual(angle(vec) * 180 / np.pi, deg)

    def test_length(self):
        pass

    def test_dot(self):
        pass

    def test_cross(self):
        pass

    def test_normalize(self):
        pass


class MultiAgentSimulationTest(unittest.TestCase):
    def test_simulations(self):
        gen = import_simulations()
        for simulation in gen:
            # simulation.configure_hdfstore()
            simulation.initial_update()
            simulation.update()
            self.assertTrue(True)


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


class MotionTest(unittest.TestCase):
    pass


class NavigationTest(unittest.TestCase):
    pass


class OrientationTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
