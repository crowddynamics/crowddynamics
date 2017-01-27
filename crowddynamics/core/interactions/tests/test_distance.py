import hypothesis.strategies as st
import numpy as np
from hypothesis import given

from crowddynamics.core.interactions.distance import distance_circle_circle, \
    distance_three_circle, distance_circle_line, distance_three_circle_line, \
    overlapping_circle_circle, overlapping_three_circle
from crowddynamics.core.vector2D.vector2D import length
from crowddynamics.testing import vector, real


@given(
    x0=vector(elements=real(-10, 10)),
    r0=real(min_value=0, max_value=1),
    x1=vector(elements=real(-10, 10)),
    r1=real(min_value=0, max_value=1)
)
def test_distance_circle_circle(x0, r0, x1, r1):
    h, n = distance_circle_circle(x0, r0, x1, r1)
    x = x0 - x1
    r_tot = r0 + r1

    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)
    assert n.dtype.type is np.float64

    assert h >= -r_tot

    if np.all(x == 0.0):
        assert np.isclose(length(n), 0.0)
    else:
        assert np.isclose(length(n), 1.0)


@given(
    x0=st.tuples(*3 * [vector(elements=real(-10, 10))]),
    r0=st.tuples(*3 * [real(min_value=0, max_value=1)]),
    x1=st.tuples(*3 * [vector(elements=real(-10, 10))]),
    r1=st.tuples(*3 * [real(min_value=0, max_value=1)])
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
    x=vector(elements=real(-10, 10)),
    r=real(min_value=0, max_value=1),
    p=vector(shape=(2, 2), elements=real(-10, 10))
)
def test_distance_circle_line(x, r, p):
    h, n = distance_circle_line(x, r, p)
    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)
    assert n.dtype.type is np.float64


@given(
    x=st.tuples(*3 * [vector(elements=real(-10, 10))]),
    r=st.tuples(*3 * [real(min_value=0, max_value=1)]),
    p=vector(shape=(2, 2), elements=real(-10, 10))
)
def test_distance_three_circle_line(x, r, p):
    h, n, r_moment = distance_three_circle_line(x, r, p)
    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)
    assert n.dtype.type is np.float64
    assert isinstance(r_moment, np.ndarray)
    assert r_moment.dtype.type is np.float64


@given(
    x1=vector(shape=(5, 2), elements=real(-10, 10)),
    r1=vector(shape=5, elements=real(min_value=0, max_value=1)),
    x2=vector(shape=2, elements=real(-10, 10)),
    r2=real(min_value=0, max_value=1)
)
def test_overlapping_circle_circle(x1, r1, x2, r2):
    flag = overlapping_circle_circle(x1, r1, x2, r2)
    assert isinstance(flag, bool)


@given(
    x1=st.tuples(*3 * [vector(shape=(5, 2), elements=real(-10, 10))]),
    r1=st.tuples(*3 * [vector(shape=5, elements=real(min_value=0, max_value=1))]),
    x2=st.tuples(*3 * [vector(shape=2, elements=real(-10, 10))]),
    r2=st.tuples(*3 * [real(min_value=0, max_value=1)])
)
def test_overlapping_three_circle(x1, r1, x2, r2):
    flag = overlapping_three_circle(x1, r1, x2, r2)
    assert isinstance(flag, bool)
