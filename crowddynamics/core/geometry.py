"""Functions for manipulating Shapely geometry objects

References:
    - http://toblerity.org/shapely/manual.html
"""
from collections import Iterable
from itertools import chain
from typing import Tuple, Iterator, Callable

import numpy as np
import skimage.draw
from matplotlib.path import Path
from shapely import speedups
from shapely.geometry import Polygon, LineString, Point, LinearRing
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from crowddynamics.core.structures.obstacles import obstacle_type_linear

if speedups.available:
    speedups.enable()


PointPair = Tuple[float, float]
LineSegment = Tuple[PointPair, PointPair]


def geom_to_linesegment(geom: BaseGeometry) -> Iterator[LineSegment]:
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
    segments = list(geom_to_linesegment(geom))
    return np.array(segments, dtype=obstacle_type_linear)


def geom_to_skimage(geom: BaseGeometry,
                    indicer: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """Convert geom

    Args:
        geom: 
        indicer: Function that converts points to indices of a discrete grid. 

    Yields:
        (np.ndarray, np.ndarray): Returned indices are in matrix order 
            (row, column) aka (y, x).
    """
    if isinstance(geom, Point):
        pass
    elif isinstance(geom, LineString):
        for line in geom_to_linesegment(geom):
            r0, c0, r1, c1 = indicer(line).flatten()
            yield skimage.draw.line(r0, c0, r1, c1)
    elif isinstance(geom, Polygon):
        i = indicer(geom.exterior)
        yield skimage.draw.polygon(i[:, 0], i[:, 1])
    elif isinstance(geom, BaseMultipartGeometry):
        for gen in (geom_to_skimage(geo, indicer) for geo in geom):
            yield from gen
    else:
        raise TypeError


def coding(ob):
    # The codes will be all "LINETO" commands, except for "MOVETO"s at the
    # beginning of each subpath
    n = len(ob.coords)
    vals = np.full(shape=n, fill_value=Path.LINETO, dtype=Path.code_type)
    vals[0] = Path.MOVETO
    return vals


def geom_to_mpl(geom: BaseGeometry) -> Path:
    """Convert geom to matplotlib Path
    
    References:
        https://bitbucket.org/sgillies/descartes/src/ddcb7f8f0542758ebf28fee5cce946413c4b2e3c/descartes/patch.py?at=default&fileviewer=file-view-default
    """
    def inner(geom: BaseGeometry) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(geom, Point):
            raise ValueError('Points')
        elif isinstance(geom, LineString):
            vertices = np.asarray(geom)[:, :2]
            codes = coding(geom)
            yield vertices, codes
        elif isinstance(geom, Polygon):
            vertices = np.concatenate([np.asarray(geom.exterior)[:, :2]] +
                                      [np.asarray(r)[:, :2] for r in geom.interiors])
            codes = np.concatenate([coding(geom.exterior)] +
                                   [coding(r) for r in geom.interiors])
            yield vertices, codes
        elif isinstance(geom, BaseMultipartGeometry):
            for gen in map(inner, geom):
                yield from gen
        else:
            raise TypeError('Argument is not subclass of {}'.format(BaseGeometry))

    _vertices = []
    _codes = []
    for vertices, codes in inner(geom):
        _vertices.append(vertices)
        _codes.append(codes)
    return Path(np.concatenate(_vertices), np.concatenate(_codes))
