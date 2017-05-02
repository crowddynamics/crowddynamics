import pytest
from hypothesis import assume
from hypothesis.core import given
import numpy as np

from crowddynamics.core.vector2D import length
from crowddynamics.testing import real
from crowddynamics.core.steering.herding import herding


@given(real(-1.0, 1.0, shape=(3, 2)))
def test_herding(e0_neigh):
    ans = herding(e0_neigh)
    assert True
