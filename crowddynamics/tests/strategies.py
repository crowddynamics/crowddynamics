"""
Module to generate various input values for testing code.
"""
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import ArrayStrategy, arrays
from shapely.geometry import Polygon


class UnitVectorStrategy(ArrayStrategy):
    def do_draw(self, data):
        if self.array_size != 2:
            raise NotImplementedError
        phi = self.element_strategy.do_draw(data)
        return np.array((np.cos(phi), np.sin(phi)), dtype=self.dtype)


def real():
    return st.floats(None, None, False, False)


def positive(include_zero=True):
    strategy = st.floats(0, None, False, False)
    if include_zero:
        return strategy
    else:
        return strategy.filter(lambda x: x != 0.0)


def vector(dim=2, elements=real()):
    return arrays(np.float64, dim, elements)


@st.composite
def vectors(draw, elements=real(), maxsize=100, dim=2):
    size = draw(st.integers(1, maxsize))
    values = draw(arrays(np.float64, (size, dim), elements))
    return values


def unit_vector(start=0, end=2 * np.pi, dim=2):
    phi = st.floats(start, end, False, False)
    return UnitVectorStrategy(phi, dim, np.float64)


def line(dim=2):
    return arrays(np.float64, (dim, dim), st.floats(None, None, False, False))


def three_vectors():
    return st.tuples(*(vector() for _ in range(3)))


def three_positive():
    return st.tuples(*(positive() for _ in range(3)))


@st.composite
def polygons(draw, low=0.0, high=1.0, max_verts=5):
    """
    Generate a random polygon.

    Args:
        draw:
        low:
        high:
        max_verts:

    Returns:
        Polygon: Random convex polygon
    """
    # TODO: buffer
    points = draw(vectors(elements=st.floats(low, high, False, False),
                          maxsize=max_verts))
    polygon = Polygon(points).convex_hull
    return polygon
