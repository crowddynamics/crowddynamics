import random

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.qhull import QhullError
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon

from crowddynamics.core.sampling import polygon_sample
from crowddynamics.simulation.agents import AgentType


def reals(min_value=None,
          max_value=None,
          exclude_zero=None,
          shape=None,
          dtype=np.float):
    """Real number strategy that excludes nan and inf.

    Args:
        min_value (Number, optional):
        max_value (Number, optional):
        exclude_zero (str, optional):
            Choices from: (None, 'exact', 'near')
        shape (int|tuple, optional):
            None for scalar output and int or tuple of int for array output.
        dtype (numpy.float):
            Numpy float type
    """
    # TODO: size as strategy
    assert dtype is None or np.dtype(dtype).kind == u'f'
    if min_value is not None and max_value is not None:
        assert max_value > min_value

    elements = st.floats(min_value, max_value, False, False)

    # Filter values
    if exclude_zero == 'exact':
        elements = elements.filter(lambda x: x != 0.0)
    elif exclude_zero == 'near':
        elements = elements.filter(lambda x: not np.isclose(x, 0.0))

    # Strategy
    if shape is None:
        return elements
    else:
        return arrays(dtype, shape, elements)


@st.composite
def unit_vectors(draw, min_angle=0, max_angle=2 * np.pi):
    phi = draw(st.floats(min_angle, max_angle, False, False))
    return np.array((np.cos(phi), np.sin(phi)), dtype=np.float64)


@st.composite
def points(draw, min_value=None, max_value=None, exclude_zero=None):
    """Strategy that generates shapely points"""
    elements = reals(min_value, max_value, exclude_zero, shape=2)
    return Point(draw(elements))


@st.composite
def linestrings(draw, min_value=None, max_value=None, exclude_zero=None,
                num_verts=2, closed=False):
    """Strategy that generates shapely linestrings"""
    assert num_verts >= 2
    elements = reals(min_value, max_value, exclude_zero, shape=(num_verts, 2))
    array = draw(elements)
    if closed:
        return LineString(np.concatenate((array, array[0].reshape((1, 2)))))
    else:
        return LineString(array)


@st.composite
def polygons(draw, min_value=None, max_value=None, exclude_zero=None,
             num_verts=3, has_holes=False):
    """Strategy that generates shapely polygons"""
    assert num_verts >= 3
    elements = reals(min_value, max_value, exclude_zero, shape=(num_verts, 2))
    shell = draw(elements)
    if has_holes:
        sample = polygon_sample(shell)
        try:
            hole = np.stack([next(sample) for _ in range(random.randint(3, 5))])
            holes = [hole]
        except QhullError:
            holes = None
        return Polygon(shell=shell, holes=holes)
    else:
        return Polygon(shell=shell)


def multipart_geometries():
    pass


@st.composite
def agents(draw, size_strategy, agent_type, attributes):
    """Agent search strategy

    Args:
        size_strategy (SearchStrategy):
            Strategy that generated integers
        agent_type (type):
            Subclass of AgentType
        attributes (dict):
            Dictionary of attribute strategies.

    Returns:
        SearchStrategy:
    """
    assert issubclass(agent_type, AgentType) and agent_type is not AgentType

    size = draw(size_strategy)
    array = np.zeros(size, agent_type.dtype())
    for i in range(size):
        agent = agent_type(**{
            name: draw(strategy) for name, strategy in attributes.items()})
        array[i] = np.array(agent)

    return array


def obstacles():
    pass
