"""Functions for manipulating Shapely geometry objects

References:
    - http://toblerity.org/shapely/manual.html
"""
from collections import Iterable
from functools import reduce
from itertools import chain
from typing import Callable

import numpy as np
import shapely.geometry as geometry
import skimage.draw
from shapely import speedups
from shapely.geometry import Polygon, LineString, Point
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from crowddynamics.core.structures import obstacle_type_linear

if speedups.available:
    speedups.enable()


class GeomTypes(object):
    POINT = 0.0
    LINESTRING = 1.0
    POLYGON_HOLE = 2.0
    POLYGON_SHELL = 3.0


def _geom_to_array(geom: BaseGeometry):
    if isinstance(geom, geometry.Point):
        yield np.array([(np.nan, GeomTypes.POINT)])
        yield np.asarray(geom.coords)
    elif isinstance(geom, geometry.LineString):
        yield np.array([(np.nan, GeomTypes.LINESTRING)])
        yield np.asarray(geom.coords)
    elif isinstance(geom, geometry.Polygon):
        for interior in geom.interiors:
            yield np.array([(np.nan, GeomTypes.POLYGON_HOLE)])
            yield np.asarray(interior)
        yield np.array([(np.nan, GeomTypes.POLYGON_SHELL)])
        yield np.asarray(geom.exterior)
    elif isinstance(geom, BaseMultipartGeometry):
        return chain.from_iterable(map(geom_to_array, geom))
    else:
        raise TypeError


def geom_to_array(geom: BaseGeometry):
    """Breaking geometry object into continuous array where objects are
     separated by array of elements (np.nan, FLAG)

    Args:
        geom:

    Returns:

    """
    return np.concatenate(list(_geom_to_array(geom)))


def geom_to_linesegment(geom: BaseGeometry):
    """Converts shapes to point pairs.

    >>> ls = LineString([(1, 2), (3, 4)])
    >>> list(geom_to_linesegment(ls))
    [((1.0, 2.0), (3.0, 4.0))]
    >>> poly = Polygon([(5, 6), (7, 8), (9, 10)])
    >>> list(geom_to_linesegment(poly))
    [((5.0, 6.0), (7.0, 8.0)),
     ((7.0, 8.0), (9.0, 10.0)),
     ((9.0, 10.0), (5.0, 6.0))]
    >>> list(geom_to_linesegment(ls | poly))
    [((1.0, 2.0), (3.0, 4.0)),
     ((5.0, 6.0), (7.0, 8.0)),
     ((7.0, 8.0), (9.0, 10.0)),
     ((9.0, 10.0), (5.0, 6.0))]

    Args:
        geom (BaseGeometry): BaseGeometry type.

    Returns:
        Iterable[LineSegment]: Iterable of linesegments

    """
    if isinstance(geom, Point):
        return iter(())
    elif isinstance(geom, LineString):
        return zip(geom.coords[:-1], geom.coords[1:])
    elif isinstance(geom, Polygon):
        return zip(geom.exterior.coords[:-1], geom.exterior.coords[1:])
    elif isinstance(geom, BaseMultipartGeometry):
        return chain.from_iterable(map(geom_to_linesegment, geom))
    else:
        raise TypeError('Argument is not subclass of {}'.format(BaseGeometry))


def geom_to_linear_obstacles(geom):
    """Converts shape(s) to array of linear obstacles."""
    segments = [] if geom is None else list(geom_to_linesegment(geom))
    return np.array(segments, dtype=obstacle_type_linear)


def draw_geom(geom: BaseGeometry,
              grid,
              indicer: Callable,
              value):
    """Draw geom to grid"""
    if isinstance(geom, Point):
        pass
    elif isinstance(geom, LineString):
        for line in geom_to_linesegment(geom):
            r0, c0, r1, c1 = indicer(line).flatten()
            x, y = skimage.draw.line(r0, c0, r1, c1)
            grid[y, x] = value
    elif isinstance(geom, Polygon):
        i = indicer(geom.exterior)
        x, y = skimage.draw.polygon(i[:, 0], i[:, 1])
        grid[y, x] = value
        x, y = skimage.draw.polygon_perimeter(i[:, 0], i[:, 1])
        grid[y, x] = value
        for j in map(indicer, geom.interiors):
            x, y = skimage.draw.polygon(j[:, 0], j[:, 1])
            grid[y, x] = 0
    elif isinstance(geom, BaseMultipartGeometry):
        for geo in geom:
            draw_geom(geo, grid, indicer, value)
    else:
        raise TypeError


def union(*geoms):
    """Union of geometries"""
    return reduce(lambda x, y: x | y, geoms)
