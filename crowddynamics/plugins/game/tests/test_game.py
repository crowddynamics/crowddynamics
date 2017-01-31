import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, assume
from hypothesis.extra.numpy import arrays
from crowddynamics.testing.strategies import real

from crowddynamics.plugins.game.game import poisson_clock, poisson_timings, \
    payoff, agent_closer_to_exit, narrow_exit_capacity


@given(
    interval=real(0.001, 0.01),
    dt=real(0.001, 0.01),
)
def test_poisson_clock(interval, dt):
    # Limit the ratio so test doesn't run forever
    assume(1/100 < interval / dt < 100)

    time_prev = 0.0
    for time in poisson_clock(interval, dt):
        assert isinstance(time, float)
        assert time_prev < time < dt
        time_prev = time


@given(
    players=real(0, 10**7, shape=10, dtype=int),
    interval=real(0.001, 0.01),
    dt=real(0.001, 0.01),
)
def test_poisson_timings(players, interval, dt):
    # Limit the ratio so test doesn't run forever
    assume(1 / 100 < interval / dt < 100)

    for index in poisson_timings(players, interval, dt):
        assert isinstance(index, int)
        assert index in players


@given(
    s_i=st.integers(0, 1),
    s_j=st.integers(0, 1),
    t_aset=real(0),
    t_evac_i=real(0),
    t_evac_j=real(0),
)
def test_payoff(s_i, s_j, t_aset, t_evac_i, t_evac_j):
    value = payoff(s_i, s_j, t_aset, t_evac_i, t_evac_j)
    assert isinstance(value, float)
    assert value >= -1.0


@given(
    points=real(shape=(2, 2)),
    position=real(shape=(10, 2)),
)
def test_agent_closer_to_exit(points, position):
    indices = agent_closer_to_exit(points, position)
    assert isinstance(indices, np.ndarray)
    assert indices.dtype.type is np.int64


@given(
    d_door=real(0.0, 3.0, exclude_zero='near'),
    d_agent=real(0.0, 3.0, exclude_zero='near'),
    d_layer=real(0.0, 1.0, exclude_zero='near') | st.none(),
    coeff=real(0.0, 3.0, exclude_zero='near')
)
def test_narrow_exit_capacity(d_door, d_agent, d_layer, coeff):
    assume(d_door > d_agent)
    capacity = narrow_exit_capacity(d_door, d_agent, d_layer, coeff)
    assert isinstance(capacity, float)
    assert capacity >= 0.0


@pytest.mark.skip
def test_best_response_strategy():
    assert True


@pytest.mark.skip
def test_egress_game():
    assert True
