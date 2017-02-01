from hypothesis import given, assume

from crowddynamics.core.random.random import poisson_clock, poisson_timings
from crowddynamics.testing import real


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