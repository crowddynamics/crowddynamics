"""
Module to generate various input values for testing code using ``Hypothesis``
library [#]_.

.. [#] https://hypothesis.readthedocs.io/en/latest/data.html?highlight=example
"""
import random
from itertools import chain

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from shapely.geometry import LineString


def real(min_value=None, max_value=None, exclude_zero=None, shape=None,
         dtype=float):
    """Real number strategy using 64-bit floating point numbers excluding
    ``nan`` and ``inf``.

    Args:
        min_value (Number, optional):
        max_value (Number, optional):
        exclude_zero (str, optional):
            Choices: (None, 'exact', 'near')
        shape (int|tuple, optiona):
            - None: Scalar output
            - int|tuple: Vector output of shape
        dtype (type):
            Choice
            - float: Floats
            - int: Integers

    Todo:
        - size as strategy
        - filtering values with assume?
    """
    if dtype is float:
        dtype = np.float64
        elements = st.floats(min_value, max_value, False, False)
    elif dtype is int:
        dtype = np.int64
        elements = st.integers(min_value, max_value)
    else:
        raise Exception("")

    # Filter values
    if exclude_zero == 'exact':
        elements = elements.filter(lambda x: x != 0.0)

    if exclude_zero == 'near':
        elements = elements.filter(lambda x: not np.isclose(x, 0.0))

    # Strategy
    if isinstance(shape, (int, tuple)):
        return arrays(dtype, shape, elements)
    else:
        return elements


@st.composite
def unit_vector(draw, start=0, end=2 * np.pi):
    phi = draw(st.floats(start, end, False, False))
    return np.array((np.cos(phi), np.sin(phi)), dtype=np.float64)


@st.composite
def polygon(draw, a=-1.0, b=1.0, num_points=5, buffer=real(0.1, 0.2)):
    r"""Generate a random polygon. Polygon should have area > 0.

    Args:
        draw:
        a (float):
            Minimum value of the bounding box

        b (float):
           Maximum value of the bounding box

        num_points (int):
        buffer (SearchStrategy|Number):
        convex_hull (bool):

    Returns:
        Polygon: Random polygon

    """
    # TODO: input strategies
    c = draw(buffer)
    points = draw(arrays(np.float64, (num_points, 2),
                         real(a + c, b - c, exclude_zero='near')))
    lines = LineString(points)
    poly = lines.buffer(c)
    # TODO: assume poly.area
    # assume(poly.area > 0)
    return poly


@st.composite
def field(draw,
          domain_strategy=polygon(a=-10.0, b=10.0, buffer=st.just(1.0)),
          target_length_strategy=st.just(1.0)):
    r"""SearchStrategy that generates ``domain``, ``targets`` and ``obstacles``
    for testing.

    Domain is created as polygon, then targets are chosen from the ``domain.exterior``
    of the polygon and rest of the exterior and ``domain.interior`` is made to
    obstacle.

    Args:
        draw:
        domain_strategy:
        target_length_strategy:

    Returns:
        (Polygon, BaseGeometry, BaseGeometry):
            - ``domain``
            - ``targets``
            - ``obstacles``
    """
    target_length = draw(target_length_strategy)
    domain = draw(domain_strategy)
    targets = LineString()
    obstacles = domain.exterior
    for obs in domain.interiors:
        obstacles |= obs

    # Select a continuous line segment from the domain.exterior as a target.
    def shifted_range(start, stop):
        shift = random.randrange(start, stop)
        return chain(range(shift, stop), range(start, shift))

    coords = domain.exterior.coords
    for i in shifted_range(0, len(coords) - 1):
        targets |= LineString([coords[i], coords[i+1]])
        if targets.length > target_length:
            break

    obstacles -= targets
    return domain, targets, obstacles
