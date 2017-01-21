import numpy as np
from hypothesis import given, assume

from crowddynamics.core.interactions.distance import distance_circle_circle, \
    distance_three_circle, distance_circle_line, distance_three_circle_line
from crowddynamics.core.vector2D.vector2D import length
from crowddynamics.testing import vector, three_vectors, \
    three_positive, line, real


@given(x0=vector(), r0=real(min_value=0), x1=vector(), r1=real(min_value=0))
def test_distance_circle_circle(x0, r0, x1, r1):
    h, n = distance_circle_circle(x0, r0, x1, r1)
    x = x0 - x1
    r_tot = r0 + r1

    assume(not np.allclose(x, 0.0))

    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)
    assert h >= -r_tot

    if np.all(x == 0.0):
        assert np.isclose(length(n), 0.0)
    else:
        assert np.isclose(length(n), 1.0)


@given(x0=three_vectors(), r0=three_positive(),
       x1=three_vectors(), r1=three_positive(), )
def test_distance_three_circle(x0, r0, x1, r1):
    h, n, r_moment0, r_moment1 = distance_three_circle(x0, r0, x1, r1)

    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)
    assert isinstance(r_moment0, np.ndarray)
    assert isinstance(r_moment1, np.ndarray)


@given(x=vector(), r=real(min_value=0), p=line(), )
def test_distance_circle_line(x, r, p):
    h, n = distance_circle_line(x, r, p)
    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)


@given(x=three_vectors(), r=three_positive(), p=line(), )
def test_distance_three_circle_line(x, r, p):
    h, n, r_moment = distance_three_circle_line(x, r, p)
    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)
    assert isinstance(r_moment, np.ndarray)


def test_overlapping_circle_circle():
    assert True


def test_overlapping_three_circle():
    assert True
