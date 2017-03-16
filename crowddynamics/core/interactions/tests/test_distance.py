import hypothesis.strategies as st
import numpy as np
from hypothesis import given

import crowddynamics.testing
from crowddynamics.core.interactions.distance import distance_circle_circle, \
    distance_three_circle, distance_circle_line, distance_three_circle_line, \
    overlapping_circle_circle, overlapping_three_circle


@given(
    x0=crowddynamics.testing.real(-10, 10, shape=2),
    r0=crowddynamics.testing.real(0.0, 1.0),
    x1=crowddynamics.testing.real(-10, 10, shape=2),
    r1=crowddynamics.testing.real(0.0, 1.0)
)
def test_distance_circle_circle(x0, r0, x1, r1):
    h, n = distance_circle_circle(x0, r0, x1, r1)
    x = x0 - x1
    r_tot = r0 + r1

    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)
    assert n.dtype.type is np.float64

    assert h >= -r_tot


@given(
    x0=st.tuples(*3 * [crowddynamics.testing.real(-10, 10, shape=2)]),
    r0=st.tuples(*3 * [crowddynamics.testing.real(0.0, 1.0)]),
    x1=st.tuples(*3 * [crowddynamics.testing.real(-10, 10, shape=2)]),
    r1=st.tuples(*3 * [crowddynamics.testing.real(0.0, 1.0)])
)
def test_distance_three_circle(x0, r0, x1, r1):
    h, n, r_moment0, r_moment1 = distance_three_circle(x0, r0, x1, r1)

    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)
    assert n.dtype.type is np.float64
    assert isinstance(r_moment0, np.ndarray)
    assert r_moment0.dtype.type is np.float64
    assert isinstance(r_moment1, np.ndarray)
    assert r_moment1.dtype.type is np.float64


@given(
    x=crowddynamics.testing.real(-10, 10, shape=2),
    r=crowddynamics.testing.real(min_value=0, max_value=1),
    p=crowddynamics.testing.real(-10, 10, shape=(2, 2))
)
def test_distance_circle_line(x, r, p):
    h, n = distance_circle_line(x, r, p)
    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)
    assert n.dtype.type is np.float64


@given(
    x=st.tuples(*3 * [crowddynamics.testing.real(-10, 10, shape=2)]),
    r=st.tuples(*3 * [crowddynamics.testing.real(min_value=0, max_value=1)]),
    p=crowddynamics.testing.real(-10, 10, shape=(2, 2))
)
def test_distance_three_circle_line(x, r, p):
    h, n, r_moment = distance_three_circle_line(x, r, p)
    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)
    assert n.dtype.type is np.float64
    assert isinstance(r_moment, np.ndarray)
    assert r_moment.dtype.type is np.float64


@given(
    agent=crowddynamics.testing.agent(5),
    x2=crowddynamics.testing.real(-10, 10, shape=2),
    r2=crowddynamics.testing.real(min_value=0, max_value=1)
)
def test_overlapping_circle_circle(agent, x2, r2):
    flag = overlapping_circle_circle(agent, agent.indices(), x2, r2)
    assert isinstance(flag, bool)


@given(
    agent=crowddynamics.testing.agent(5),
    x2=st.tuples(*3 * [crowddynamics.testing.real(-10, 10, shape=2)]),
    r2=st.tuples(*3 * [crowddynamics.testing.real(min_value=0, max_value=1)])
)
def test_overlapping_three_circle(agent, x2, r2):
    flag = overlapping_three_circle(agent, agent.indices(), x2, r2)
    assert isinstance(flag, bool)
