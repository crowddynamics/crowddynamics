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
