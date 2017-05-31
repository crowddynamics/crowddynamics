import hypothesis.strategies as st
import numpy as np
from hypothesis import given, assume

from crowddynamics.core.evacuation import agent_closer_to_exit, \
    narrow_exit_capacity
from crowddynamics.testing import reals


@given(c_door=reals(shape=2),
       position=reals(shape=(10, 2)))
def test_agent_closer_to_exit(c_door, position):
    indices = agent_closer_to_exit(c_door, position)
    assert isinstance(indices, np.ndarray)
    assert indices.dtype.type is np.int64


@given(d_door=reals(0.0, 3.0, exclude_zero='near'),
       d_agent=reals(0.0, 3.0, exclude_zero='near'),
       d_layer=reals(0.0, 1.0, exclude_zero='near') | st.none(),
       coeff=reals(0.0, 3.0, exclude_zero='near'))
def test_narrow_exit_capacity(d_door, d_agent, d_layer, coeff):
    assume(d_door >= d_agent)
    capacity = narrow_exit_capacity(d_door, d_agent, d_layer, coeff)
    assert isinstance(capacity, float)
    assert capacity >= 0.0
