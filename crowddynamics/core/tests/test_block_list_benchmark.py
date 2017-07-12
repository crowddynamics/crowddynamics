import random
from collections import defaultdict

import numpy as np
import pytest

from crowddynamics.core.block_list import MutableBlockList


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
        mutable_blocklist = MutableBlockList(cell_size)
        for key, value in zip(keys, values):
            mutable_blocklist[key] = value

    benchmark(f)
    assert True


@pytest.mark.parametrize('cell_size', (0.27,))
@pytest.mark.parametrize('size', (100, 500, 1000))
def test_mutable_blocklist_getitem(benchmark, size, cell_size):
    mutable_blocklist = MutableBlockList(cell_size)

    for value in range(size):
        key = np.random.uniform(-1.0, 1.0, 2)
        mutable_blocklist[key] = value

    key = np.random.uniform(-1.0, 1.0, 2)
    benchmark(mutable_blocklist.__getitem__, key)
    assert True
