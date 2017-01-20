import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, assume
from hypothesis.extra.numpy import arrays

from crowddynamics.plugins.game.game import poisson_clock, poisson_timings, \
    payoff, agent_closer_to_exit, exit_capacity


@given(
    interval=st.floats(0.001, 0.01, allow_nan=False, allow_infinity=False),
    dt=st.floats(0.001, 0.01, allow_nan=False, allow_infinity=False),
)
def test_poisson_clock(interval, dt):
    # Limit the ratio so test doesn't run forever
    assume(1/100 < interval / dt < 100)

    times = [time for time in poisson_clock(interval, dt)]
    assert isinstance(times, list)
    assert np.all(np.array(times) < dt)
    assert sorted(times) == times


@given(
    players=arrays(np.int64, 10, st.integers(0, 10**7)),
    interval=st.floats(0.001, 0.01, allow_nan=False, allow_infinity=False),
    dt=st.floats(0.001, 0.01, allow_nan=False, allow_infinity=False),
)
def test_timings(players, interval, dt):
    # Limit the ratio so test doesn't run forever
    assume(1 / 100 < interval / dt < 100)

    indices = poisson_timings(players, interval, dt)
    assert isinstance(indices, list)


@given(
    s_i=st.integers(0, 1),
    s_j=st.integers(0, 1),
    t_aset=st.floats(0, allow_nan=False, allow_infinity=False),
    t_evac_i=st.floats(0, allow_nan=False, allow_infinity=False),
    t_evac_j=st.floats(0, allow_nan=False, allow_infinity=False),
)
def test_payoff(s_i, s_j, t_aset, t_evac_i, t_evac_j):
    value = payoff(s_i, s_j, t_aset, t_evac_i, t_evac_j)
    assert isinstance(value, float)
    assert value >= -1.0


@given(
    points=arrays(np.float64, (2, 2), st.floats(allow_nan=False, allow_infinity=False)),
    position=arrays(np.float64, (10, 2), st.floats(allow_nan=False, allow_infinity=False)),
)
def test_agent_closer_to_exit(points, position):
    indices = agent_closer_to_exit(points, position)
    assert isinstance(indices, np.ndarray)
    assert indices.dtype.type is np.int64


@given(
    points=arrays(np.float64, (2, 2), st.floats(allow_nan=False, allow_infinity=False)),
    agent_radius=st.floats(min_value=0.0, allow_nan=False, allow_infinity=False)
)
def test_exit_capacity(points, agent_radius):
    assume(agent_radius != 0.0)
    capacity = exit_capacity(points, agent_radius)
    assert capacity >= 0.0


@pytest.mark.skip
def test_best_response_strategy():
    assert True


@pytest.mark.skip
def test_egress_game():
    assert True
