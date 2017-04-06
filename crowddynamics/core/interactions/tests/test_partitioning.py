import random
from collections import defaultdict

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from crowddynamics.core.interactions.partitioning import block_list, \
    MutableBlockList
from crowddynamics.strategies import real


@given(points=real(-10.0, 10.0, shape=(10, 2)),
       cell_size=st.floats(0.1, 1.0))
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


@pytest.mark.parametrize('cell_size', (0.27,))
@pytest.mark.parametrize('size', (100, 250, 500, 1000, 10000))
def test_blocklist_benchmark(benchmark, size, cell_size):
    points = np.random.uniform(-1.0, 1.0, (size, 2))
    benchmark(block_list, points, cell_size)
    assert True


@pytest.mark.parametrize('cell_size', (0.27,))
@pytest.mark.parametrize('size', (100, 250, 500, 1000))
def test_defaultdict_setitem(benchmark, size, cell_size):
    def f():
        keys = ((random.uniform(-1.0, 1.0) // cell_size,
                 random.uniform(-1.0, 1.0) // cell_size) for _ in range(size))
        values = range(size)
        d = defaultdict(list)
        for key, value in zip(keys, values):
            d[key].append(value)

    benchmark(f)
    assert True


@pytest.mark.parametrize('cell_size', (0.27,))
@pytest.mark.parametrize('size', (100, 250, 500, 1000))
def test_mutable_blocklist_setitem(benchmark, size, cell_size):
    def f():
        keys = ((random.uniform(-1.0, 1.0),
                 random.uniform(-1.0, 1.0)) for _ in range(size))
        values = range(size)
        mutable_blocklist = MutableBlockList(cell_size, radius=1)
        for key, value in zip(keys, values):
            mutable_blocklist[key] = value

    benchmark(f)
    assert True


@pytest.mark.parametrize('cell_size', (0.27,))
@pytest.mark.parametrize('size', (100, 500, 1000))
def test_mutable_blocklist_getitem(benchmark, size, cell_size):
    mutable_blocklist = MutableBlockList(cell_size, radius=1)

    for value in range(size):
        key = np.random.uniform(-1.0, 1.0, 2)
        mutable_blocklist[key] = value

    key = np.random.uniform(-1.0, 1.0, 2)
    benchmark(mutable_blocklist.__getitem__, key)
    assert True


def test_blocklist_compare():
    pass
