"""Functions for manipulating Shapely geometry objects

References:
    - http://toblerity.org/shapely/manual.html
"""
from collections import Iterable

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


def check_shapes(shapes, types):
    """Checks type and flattens shapes.

    Args:
        shapes (shapely.geometry.base.BaseGeometry):
            Shapes
        types (shapely.geometry.base.BaseGeometry):
            Allowed types of subclass shapely.BaseGeometry

    Returns:
        List: List of shapes
    """

    def _set_shape(_shapes, _types, _coll):
        if isinstance(_shapes, Iterable):
            for shape in _shapes:
                _set_shape(shape, _types, _coll)
        elif isinstance(_shapes, _types):
            _coll.append(_shapes)
        elif _shapes is None:
            pass
        else:
            raise ValueError("shape {} not in types {}".format(_shapes, _types))

    # TODO: Geometry Collection?
    coll = []
    _set_shape(shapes, types, coll)
    return coll


def shapes_to_point_pairs(shapes):
    """Converts shapes to pairs of points representing the line segments of the
    shapes.

    Args:
        shapes (shapely.geometry.base.BaseGeometry): Shapes

    Returns:
        numpy.ndarray: Numpy array of point pairs.
    """

    def _shapes_to_points(_shapes, _points):
        if isinstance(_shapes, Iterable):
            for shape in _shapes:
                _shapes_to_points(shape, _points)
        elif isinstance(_shapes, Polygon):
            _shapes_to_points(_shapes.exterior, _points)
        elif isinstance(_shapes, BaseGeometry):
            a = np.asarray(_shapes)
            for i in range(len(a) - 1):
                _points.append((a[i], a[i + 1]))
        elif _shapes is None:
            pass
        else:
            raise ValueError("")

    points = []
    _shapes_to_points(shapes, points)
    return np.array(points)
