import numpy as np
from hypothesis import given, assume

from crowddynamics.core.vector.vector2D import cross, wrap_to_pi, truncate, \
    rotate270, normalize, length, angle, rotate90, dot, unit_vector
from crowddynamics.testing import real


@given(phi=real())
def test_wrap_to_pi(phi):
    ans = wrap_to_pi(phi)
    assert isinstance(ans, float)
    assert -np.pi <= ans <= np.pi
    if (phi + np.pi) % (2 * np.pi) == 0.0:
        if phi > 0:
            assert ans == np.pi
        else:
            assert ans == -np.pi


@given(a=real(shape=2))
def test_rotate90(a):
    ans = rotate90(a)
    assert isinstance(ans, np.ndarray)


@given(a=real(shape=2))
def test_rotate270(a):
    ans = rotate270(a)
    assert isinstance(ans, np.ndarray)


@given(a=real(shape=2))
def test_angle(a):
    ans = angle(a)
    assert isinstance(ans, float)
    assert -np.pi <= ans <= np.pi


@given(a=real(shape=(10, 2)))
def test_angle_vec(a):
    ans = angle(a)
    assert isinstance(ans, np.ndarray)
    assert np.all((-np.pi <= ans) & (ans <= np.pi))


@given(a=real(shape=2))
def test_length(a):
    ans = length(a)
    assert isinstance(ans, float)
    assert ans >= 0


@given(a=real(shape=(10, 2)))
def test_length_vec(a):
    ans = length(a)
    assert isinstance(ans, np.ndarray)
    assert np.all(ans >= 0)


@given(a=real(shape=2), b=real(shape=2))
def test_dot(a, b):
    ans = dot(a, b)
    assert isinstance(ans, float)


@given(a=real(shape=2), b=real(shape=2))
def test_cross(a, b):
    ans = cross(a, b)
    assert isinstance(ans, float)


@given(a=real(shape=2))
def test_normalize(a):
    assume(not np.allclose(a, 0.0) or np.all(a == 0.0))
    assume(not length(a) > 10**8)

    ans = normalize(a)
    assert isinstance(ans, np.ndarray)
    l = length(ans)
    if np.all(a == 0.0):
        assert np.isclose(l, 0.0)
    else:
        assert np.isclose(l, 1.0)


@given(v=real(-1.0, 1.0, exclude_zero='near', shape=2),
       l=real(0.0, 1.0, exclude_zero='near'))
def test_truncate(v, l):
    vlen = length(v)

    truncate(v, l)
    if vlen > l:
        assert np.isclose(length(v), l)
    else:
        assert length(v) <= l


def test_unit_vector():
    ans = unit_vector(0.0)
    assert isinstance(ans, np.ndarray)
