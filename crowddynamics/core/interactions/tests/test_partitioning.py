import hypothesis.strategies as st
import numpy as np
from hypothesis import given

from crowddynamics.core.interactions.partitioning import block_list
from crowddynamics.testing import vectors, real


@given(
    points=vectors(elements=real(-10.0, 10.0), dim=2, maxsize=30),
    cell_width=st.floats(0.1, 1.0)
)
def test_block_list(points, cell_width):
    n, m = points.shape

    index_list, count, offset, x_min, x_max = block_list(points, cell_width)

    assert isinstance(index_list, np.ndarray)
    assert index_list.dtype.type is np.int64
    for i in range(n):
        assert i in index_list

    assert isinstance(count, np.ndarray)
    assert count.dtype.type is np.int64
    assert np.sum(count) == n

    assert isinstance(offset, np.ndarray)
    assert offset.dtype.type is np.int64
    assert 0 <= np.min(offset) <= np.max(offset) <= n
    assert np.all(np.sort(offset) == offset)

    assert isinstance(x_min, np.ndarray)
    assert x_min.dtype.type is np.int64

    assert isinstance(x_max, np.ndarray)
    assert x_max.dtype.type is np.int64
