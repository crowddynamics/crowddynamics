import numpy as np
import pytest
from hypothesis.core import given
from hypothesis.extra.numpy import arrays

from crowddynamics.core.steering.quickest_path import direction_map, \
    distance_map, meshgrid
from crowddynamics.testing import reals


@given(step=reals(0.1, 1))
def test_meshgrid(step):
    mgrid = meshgrid(step, minx=0.0, miny=0.0, maxx=10.0, maxy=10.0)
    assert True


@pytest.mark.skip
def test_distance_map(mgrid, targets, obstacles):
    dmap = distance_map(mgrid, targets, obstacles)
    assert True


@given(dmap=arrays(dtype=np.float64, shape=(5, 5),
                   elements=reals(-10, 10, exclude_zero=True)))
def test_direction_map(dmap):
    u, v = direction_map(dmap)
    assert u.shape == dmap.shape
    assert v.shape == dmap.shape
