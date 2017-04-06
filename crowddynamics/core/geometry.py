"""Functions for manipulating Shapely geometry objects

References:
    - http://toblerity.org/shapely/manual.html
"""
from collections import Iterable

from shapely import speedups
from shapely.geometry import Polygon, LineString, LinearRing


if speedups.available:
    speedups.enable()


def geom_to_pairs(geom) -> list:
    """Converts shapes to point pairs.
    
    >>> ls = LineString([(1, 2), (0, 0)])
    >>> lr = LinearRing([(1, 2), (0, 0), (2, 1)])
    >>> poly = Polygon([(1, 2), (0, 0), (2, 1)])
    >>> geom_to_pairs((ls, lr, poly))
    [((1.0, 2.0), (0.0, 0.0)),
     ((1.0, 2.0), (0.0, 0.0)),
     ((0.0, 0.0), (2.0, 1.0)),
     ((2.0, 1.0), (1.0, 2.0)),
     ((1.0, 2.0), (0.0, 0.0)),
     ((0.0, 0.0), (2.0, 1.0)),
     ((2.0, 1.0), (1.0, 2.0))]

    Args:
        shape: 
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
