import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis.core import example
from hypothesis.strategies import just

import crowddynamics.testing
from crowddynamics.core.steering.navigation import direction_map, distance_map


@pytest.mark.skip
@given(step=just(0.01), field=crowddynamics.testing.field())
@settings(max_examples=10)
def test_distance_map(step, field):
    mgrid, dmap, phi = distance_map(*field, step=step)

    assert isinstance(mgrid, list)
    for i in range(len(mgrid)):
        assert isinstance(mgrid[i], np.ndarray)
        assert mgrid[i].dtype.type is np.float64

    assert isinstance(dmap, np.ndarray)
    assert dmap.dtype.type is np.float64
    assert len(dmap.shape) == 2

    assert isinstance(phi, np.ma.MaskedArray)


@pytest.mark.skip
def test_travel_time_map():
    assert True


@pytest.mark.skip
@given(dmap=crowddynamics.testing.real(-1.0, 1.0, shape=(10, 10)))
@example(dmap=np.zeros((10, 10)))
@example(dmap=np.ones((10, 10)))
def test_direction_map(dmap):
    dir_map = direction_map(dmap)

    assert isinstance(dir_map, np.ndarray)
    assert dmap.dtype.type is np.float64
    assert len(dir_map.shape) == 3


@pytest.mark.skip
def test_merge_dir_maps():
    assert True


@pytest.mark.skip
def test_static_potential():
    assert True


@pytest.mark.skip
def test_dynamic_potential():
    assert True

