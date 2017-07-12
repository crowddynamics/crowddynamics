from array import array
from random import uniform

import pytest
from sortedcontainers.sortedlist import SortedList

from crowddynamics.core.block_list import MutableBlockList


def points(dimensions, interval=(-1.0, 1.0)):
    while True:
        yield tuple(uniform(*interval) for _ in range(dimensions))


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
