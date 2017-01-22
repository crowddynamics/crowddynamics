"""
Module to generate various input values for testing code using ``Hypothesis``
library.
"""
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from shapely.geometry import LineString
# TODO: Namedtuple agent strategy


def real(min_value=None, max_value=None, exclude_zero=None):
    """Real number strategy using 64-bit floating point numbers excluding
    ``nan`` and ``inf``.

    Args:
        min_value:
        max_value:
        exclude_zero (str, optional):
            Choices: (None, 'exact', 'near')

    """
    strategy = st.floats(min_value, max_value, False, False)
    # TODO: Maybe use assume instead?
    if exclude_zero == 'exact':
        return strategy.filter(lambda x: x != 0.0)
    elif exclude_zero == 'near':
        return strategy.filter(lambda x: not np.isclose(x, 0.0))
    return strategy


def vector(dtype=np.float64, shape=2, elements=real()):
    return arrays(dtype, shape, elements)


@st.composite
def vectors(draw, elements=real(), maxsize=100, dim=2):
    size = draw(st.integers(1, maxsize))
    values = draw(arrays(np.float64, (size, dim), elements))
    return values


@st.composite
def unit_vector(draw, start=0, end=2 * np.pi):
    phi = draw(st.floats(start, end, False, False))
    return np.array((np.cos(phi), np.sin(phi)), dtype=np.float64)


@st.composite
def polygons(draw, min_value=-1.0, max_value=1.0, num_points=5,
             convex_hull=False):
    """
    Generate a random polygon. Polygon should have area > 0.

    Args:
        draw:
        min_value (float):
        max_value (float):
        num_points (int):
        convex_hull (Boolean):

    Returns:
        Polygon: Random polygon

    """
    points = draw(arrays(np.float64, (num_points, 2),
                         real(min_value, max_value, exclude_zero='near')))
    buffer = draw(real(0.1, 0.2))
    # FIXME: Remove convex hull when sampling support None convex polygons
    if convex_hull:
        return LineString(points).buffer(buffer).convex_hull
    else:
        return LineString(points).buffer(buffer)
