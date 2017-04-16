"""Functions for manipulating Shapely geometry objects

References:
    - http://toblerity.org/shapely/manual.html
"""
from collections import Iterable
from typing import List, Tuple

from shapely import speedups
from shapely.geometry import Polygon, LineString, LinearRing
from shapely.geometry.base import BaseGeometry

if speedups.available:
    speedups.enable()


PointPair = Tuple[float, float]


def geom_to_pairs(geom: BaseGeometry) -> List[Tuple[PointPair, PointPair]]:
    """Converts shapes to point pairs.
    
    >>> geom_to_pairs([])
    []
    >>> ls = LineString([(1, 2), (3, 4)])
    >>> geom_to_pairs(ls)
    [((1.0, 2.0), (3.0, 4.0))]
    >>> lr = LinearRing([(5, 6), (7, 8), (9, 10)])
    >>> geom_to_pairs(lr)
    [((5.0, 6.0), (7.0, 8.0)),
     ((7.0, 8.0), (9.0, 10.0)),
     ((9.0, 10.0), (5.0, 6.0))]
    >>> poly = Polygon([(11, 12), (13, 14), (15, 16)])
    >>> geom_to_pairs(poly)
    [((11.0, 12.0), (13.0, 14.0)),
     ((13.0, 14.0), (15.0, 16.0)),
     ((15.0, 16.0), (11.0, 12.0))]
    >>> geom_to_pairs((ls, lr, poly))
    [((1.0, 2.0), (3.0, 4.0)),
     ((5.0, 6.0), (7.0, 8.0)),
     ((7.0, 8.0), (9.0, 10.0)),
     ((9.0, 10.0), (5.0, 6.0)),
     ((11.0, 12.0), (13.0, 14.0)),
     ((13.0, 14.0), (15.0, 16.0)),
     ((15.0, 16.0), (11.0, 12.0))]

    Args:
        geom: 
            Shape or iterable of shapes. Iterables can be nested.

    Returns:
        list: List of tuples of points pairs. 

    """
    if isinstance(geom, Iterable):
        return sum(map(geom_to_pairs, geom), [])
    elif isinstance(geom, Polygon):
        return geom_to_pairs(geom.exterior)
    elif isinstance(geom, (LineString, LinearRing)):
        return list(zip(geom.coords[:-1], geom.coords[1:]))
    else:
        return []
