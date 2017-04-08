import hypothesis.strategies as st
import numpy as np
from hypothesis import given

from crowddynamics.testing import real
from crowddynamics.core.interactions.distance import distance_circles, \
    distance_three_circles, distance_circle_line, distance_three_circle_line


@given(x0=real(-10, 10, shape=2),
       r0=real(0.0, 1.0),
       x1=real(-10, 10, shape=2),
       r1=real(0.0, 1.0))
def test_distance_circle_circle(x0, r0, x1, r1):
    h, n = distance_circles(x0, r0, x1, r1)
    x = x0 - x1
    r_tot = r0 + r1

    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)
    assert n.dtype.type is np.float64

    assert h >= -r_tot


@given(x0=st.tuples(*3 * [real(-10, 10, shape=2)]),
       r0=st.tuples(*3 * [real(0.0, 1.0)]),
       x1=st.tuples(*3 * [real(-10, 10, shape=2)]),
       r1=st.tuples(*3 * [real(0.0, 1.0)]))
def test_distance_three_circle(x0, r0, x1, r1):
    h, n, r_moment0, r_moment1 = distance_three_circles(x0, r0, x1, r1)

    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)
    assert n.dtype.type is np.float64
    assert isinstance(r_moment0, np.ndarray)
    assert r_moment0.dtype.type is np.float64
    assert isinstance(r_moment1, np.ndarray)
    assert r_moment1.dtype.type is np.float64


@given(x=real(-10, 10, shape=2),
       r=real(min_value=0, max_value=1),
       p0=real(-10, 10, shape=2),
       p1=real(-10, 10, shape=2))
def test_distance_circle_line(x, r, p0, p1):
    h, n = distance_circle_line(x, r, p0, p1)
    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)
    assert n.dtype.type is np.float64


@given(x=st.tuples(*3 * [real(-10, 10, shape=2)]),
       r=st.tuples(*3 * [real(min_value=0, max_value=1)]),
       p0=real(-10, 10, shape=2),
       p1=real(-10, 10, shape=2))
def test_distance_three_circle_line(x, r, p0, p1):
    h, n, r_moment = distance_three_circle_line(x, r, p0, p1)
    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)
    assert n.dtype.type is np.float64
    assert isinstance(r_moment, np.ndarray)
    assert r_moment.dtype.type is np.float64
