import hypothesis.strategies as st
import numpy as np
from hypothesis import given, assume
from hypothesis.extra.numpy import arrays

from crowddynamics.core.rand import poisson_clock, poisson_timings
from crowddynamics.testing import reals


@given(interval=reals(0.001, 0.01), dt=reals(0.001, 0.01))
def test_poisson_clock(interval, dt):
    # Limit the ratio so test doesn't run forever
    assume(1/100 < interval / dt < 100)

    time_prev = 0.0
    for time in poisson_clock(interval, dt):
        assert isinstance(time, float)
        assert time_prev < time < dt
        time_prev = time


@given(players=arrays(elements=st.integers(0, 10 ** 7), shape=10,
                      dtype=np.int64),
       interval=reals(0.001, 0.01),
       dt=reals(0.001, 0.01))
def test_poisson_timings(players, interval, dt):
    # Limit the ratio so test doesn't run forever
    assume(1 / 100 < interval / dt < 100)

    for index in poisson_timings(players, interval, dt):
        assert isinstance(index, int)
        assert index in players
