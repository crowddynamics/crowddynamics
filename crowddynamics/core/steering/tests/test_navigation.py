import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import just

import crowddynamics.testing.strategies as st
from crowddynamics.core.steering import distance_map


@given(step=just(0.01), field=st.field())
@settings(max_examples=10)
def test_distance_map(step, field):
    mgrid, dmap, phi = distance_map(*field, step=step)

    assert isinstance(mgrid, list)
    for i in range(len(mgrid)):
        assert isinstance(mgrid[i], np.ndarray)
        assert mgrid[i].dtype.type is np.float64

    assert isinstance(dmap, np.ndarray)
    assert dmap.dtype.type is np.float64

    assert isinstance(phi, np.ma.MaskedArray)


@pytest.mark.skip
def test_travel_time_map():
    assert True


def test_direction_map():
    assert True


def test_merge_dir_maps():
    assert True


def test_static_potential():
    assert True


@pytest.mark.skip
def test_dynamic_potential():
    assert True

