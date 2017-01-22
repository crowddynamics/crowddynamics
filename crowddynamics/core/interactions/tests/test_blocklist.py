import hypothesis.strategies as st
import numpy as np
from hypothesis import given

from crowddynamics.core.interactions.partitioning import block_list
from crowddynamics.testing import vectors


@given(points=vectors(elements=st.floats(0.0, 1.0)),
       cell_width=st.floats(0.1, 1.0))
def test_block_list(points, cell_width):
    index_list, count, offset, x_min, x_max = block_list(points, cell_width)

    assert isinstance(index_list, np.ndarray)
    assert index_list.dtype.type is np.int64

    assert isinstance(count, np.ndarray)
    assert count.dtype.type is np.int64

    assert isinstance(offset, np.ndarray)
    assert offset.dtype.type is np.int64

    assert isinstance(x_min, np.ndarray)
    assert x_min.dtype.type is np.int64

    assert isinstance(x_max, np.ndarray)
    assert x_max.dtype.type is np.int64
