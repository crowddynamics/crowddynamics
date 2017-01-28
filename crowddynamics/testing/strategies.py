"""
Module to generate various input values for testing code using ``Hypothesis``
library [#]_.

.. [#] https://hypothesis.readthedocs.io/en/latest/data.html?highlight=example
"""
from itertools import chain

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.searchstrategy.strategies import SearchStrategy
from shapely.geometry import LineString
import random

from shapely.geometry import Polygon

from crowddynamics.multiagent import Agent


def real(min_value=None, max_value=None, exclude_zero=None, shape=None):
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
    """
    dtype = np.float64
    elements = st.floats(min_value, max_value, False, False)

    # Excluded values
    # TODO: Maybe use assume instead?
    if exclude_zero == 'exact':
        return elements.filter(lambda x: x != 0.0)
    if exclude_zero == 'near':
        return elements.filter(lambda x: not np.isclose(x, 0.0))

    # Strategy
    if shape is not None:
        return arrays(dtype, shape, elements)
    else:
        return elements


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
def polygon(draw, a=-1.0, b=1.0, num_points=5, buffer=real(0.1, 0.2),
            convex_hull=False):
    r"""
    Generate a random polygon. Polygon should have area > 0.

    Args:
        draw:
        a (float):
            Minimum value of the bounding box

        b (float):
           Maximum value of the bounding box

        num_points (int):
        buffer (SearchStrategy):
        convex_hull (Boolean):

    Returns:
        Polygon: Random polygon

    """
    c = draw(buffer)
    points = draw(arrays(np.float64, (num_points, 2),
                         real(a + c, b - c, exclude_zero='near')))
    # points *= (1 - c / abs(b - a))
    lines = LineString(points)
    poly = lines.buffer(c)
    # assume(poly.area > 0)
    if convex_hull:
        return poly.convex_hull
    else:
        return poly


@st.composite
def field(draw,
          domain_strategy=polygon(),
          target_length_strategy=real(0.3, 0.5)):
    r"""
    SearchStrategy that generates a domain, targets and obstacles. Domain
    is created as polygon, then targets are chosen from the ``domain.exterior``
    of the polygon and rest of the exterior and ``domain.interior`` is made to
    obstacle.

    Args:
        draw:
        domain_strategy:
        target_length:

    Returns:
        (Polygon, MultiLineString, MultiLineString):
            - domain
            - targets
            - obstacles

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
    for i in shifted_range(0, len(coords)):
        targets |= LineString([coords[i], coords[i]])
        if targets.length < target_length:
            break

    obstacles -= targets
    return domain, targets, obstacles


@st.composite
def agent(draw, size):
    r"""
    Agent SearchStrategy

    Args:
        draw:
        size (int):

    Returns:
        SearchStrategy:

    """
    agent = Agent(size=size)

    for i in range(size):
        index = agent.add(
            draw(real(-100, 100, shape=2)),
            draw(real(1.0, 100.0)),
            draw(real(0.1, 1.0)),
            draw(real(0.1, 1.0)),
            draw(real(0.1, 1.0)),
            draw(real(0.1, 1.0)),
            draw(real()),
            draw(real()),
            draw(real(-10, 10)),
        )
        agent.set_motion(
            index,
            draw(real(-np.pi, np.pi)),
            draw(real(-100, 100, shape=2)),
            draw(real(-100, 100)),
            draw(unit_vector()),
            draw(real(-np.pi, np.pi))
        )
    return agent
