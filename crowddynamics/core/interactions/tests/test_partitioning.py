import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from crowddynamics.core.interactions.partitioning import block_list
from crowddynamics.testing import real


@given(
    points=real(-10.0, 10.0, shape=(10, 2)),
    cell_size=st.floats(0.1, 1.0)
)
def test_block_list(points, cell_size):
    n, m = points.shape

    index_list, count, offset, x_min, x_max = block_list(points, cell_size)

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


# TODO: make fixtures
cell_size = 0.001
points = np.random.uniform(-1.0, 1.0, (10000, 2))


@pytest.mark.parametrize('size', (100, 1000, 10000))
def test_blocklist_benchmark(benchmark, size):
    benchmark(block_list, points[:size], cell_size)
    assert True
