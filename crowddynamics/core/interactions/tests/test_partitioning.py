from array import array
from random import uniform

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from sortedcontainers.sortedlist import SortedList

from crowddynamics.core.interactions.partitioning import block_list, \
    MutableBlockList
from crowddynamics.testing import real


def points(dimensions, interval=(-1.0, 1.0)):
    while True:
        yield tuple(uniform(*interval) for _ in range(dimensions))


@given(points=real(-10.0, 10.0, shape=(10, 2)), cell_size=st.floats(0.1, 1.0))
def test_block_list(points, cell_size):
    n, m = points.shape

    index_list, count, offset, shape = block_list(points, cell_size)

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

    assert isinstance(shape, np.ndarray)
    assert shape.dtype.type is np.int64


@pytest.mark.parametrize('dimensions', (2, 3))
@pytest.mark.parametrize('default_list', (list, SortedList, lambda: array('i')))
def test_mutable_block_list(dimensions, default_list, cell_size=1.0):
    mutable_blocklist = MutableBlockList(cell_size, default_list)
    size = 100
    _list = default_list()
    for i, point in zip(range(size), points(dimensions, (-1.0, 2.0))):
        _list.append(i)
        mutable_blocklist[point] = i
    assert set(mutable_blocklist.nearest(dimensions * (0.0,), radius=1)) == \
           set(_list)
