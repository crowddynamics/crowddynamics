import numpy as np
from copy import deepcopy
from hypothesis import given

from crowddynamics.core.vector2D import cross2d, wrap_to_pi, truncate, \
    rotate270, normalize, length, angle, rotate90, dot2d
from crowddynamics.tests.strategies import real, vector


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


@given(a=vector())
def test_rotate90(a):
    ans = rotate90(a)
    assert isinstance(ans, np.ndarray)


@given(a=vector())
def test_rotate270(a):
    ans = rotate270(a)
    assert isinstance(ans, np.ndarray)


@given(a=vector())
def test_angle(a):
    ans = angle(a)
    assert isinstance(ans, float)
    assert -np.pi <= ans <= np.pi


@given(a=vector())
def test_length(a):
    ans = length(a)
    assert isinstance(ans, float)
    assert ans >= 0


@given(a=vector(), b=vector())
def test_dot(a, b):
    ans = dot2d(a, b)
    assert isinstance(ans, float)


@given(a=vector(), b=vector())
def test_cross(a, b):
    ans = cross2d(a, b)
    assert isinstance(ans, float)


@given(a=vector())
def test_normalize(a):
    # FIXME: floats close to zero
    ans = normalize(a)
    assert isinstance(ans, np.ndarray)
    l = length(ans)
    if np.all(a == 0.0):
        assert np.isclose(l, 0.0)
    else:
        assert np.isclose(l, 1.0)


@given(a=vector(), b=real())
def test_truncate(a, b):
    truncate(a, b)
    assert length(a) <= b
