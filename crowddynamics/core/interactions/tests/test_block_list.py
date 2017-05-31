from array import array
from random import uniform

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from hypothesis.core import example
from sortedcontainers.sortedlist import SortedList

from crowddynamics.core.interactions.block_list import block_list, \
    MutableBlockList
from crowddynamics.testing import reals


def points(dimensions, interval=(-1.0, 1.0)):
    while True:
        yield tuple(uniform(*interval) for _ in range(dimensions))


def test_block_list_nan():
    cell_size = 0.1
    points = np.array([(np.nan, 0.0),
                       (1.0, 1.0)])
    n, m = points.shape

    # Bad value in points argument causes value error
    with pytest.raises(ValueError):
        block_list(points, cell_size)


@given(points=reals(-10.0, 10.0, shape=(10, 2)),
       cell_size=st.floats(0.1, 1.0))
@example(points=np.zeros((0, 2)), cell_size=0.1)
@example(points=np.zeros((1, 2)), cell_size=0.1)
def test_block_list(points, cell_size):
    n, m = points.shape
    index_list, count, offset, shape = block_list(points, cell_size)

    for i in range(n):
        assert i in index_list
    assert np.sum(count) == n
    assert 0 <= np.min(offset) <= np.max(offset) <= n
    assert np.all(np.sort(offset) == offset)


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
