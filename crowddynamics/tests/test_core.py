import unittest
import pytest
import hypothesis.strategies as st
import numpy as np
from hypothesis import given

from crowddynamics.core.block_list import block_list
from crowddynamics.core.distance import distance_circle_circle, \
    distance_three_circle, distance_circle_line, distance_three_circle_line
from crowddynamics.core.motion import force_fluctuation, torque_fluctuation, \
    force_adjust, torque_adjust, force_social_helbing, force_contact
from crowddynamics.core.vector2D import cross2d, wrap_to_pi, truncate, \
    rotate270, normalize, length, angle, rotate90, dot2d
from crowddynamics.tests.strategies import real, positive, vector, vectors, \
    line, three_vectors, three_positive


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
    ans = normalize(a)
    assert isinstance(ans, np.ndarray)
    l = length(ans)
    # FIXME: floats close to zero
    if np.all(a == 0.0):
        assert np.isclose(l, 0.0)
    else:
        assert np.isclose(l, 1.0)


@given(a=vector(), b=real())
def test_truncate(a, b):
    truncate(a, b)
    assert length(a) <= b


@given(mass=positive(), scale=positive())
def test_force_fluctuation(mass, scale):
    ans = force_fluctuation(mass, scale)
    assert isinstance(ans, np.ndarray)


@given(inertia_rot=positive(), scale=positive())
def test_torque_fluctuation(inertia_rot, scale):
    ans = torque_fluctuation(inertia_rot, scale)
    assert isinstance(ans, np.ndarray)


@given(mass=positive(), tau_adj=positive(), v0=positive(), e0=vector(),
       v=vector())
def test_force_adjust(mass, tau_adj, v0, e0, v):
    if tau_adj == 0.0:
        with pytest.raises(ZeroDivisionError):
            force_adjust(mass, tau_adj, v0, e0, v)
    else:
        ans = force_adjust(mass, tau_adj, v0, e0, v)
        assert isinstance(ans, np.ndarray)


@given(inertia_rot=positive(), tau_rot=positive(), phi_0=real(), phi=real(),
       omega_0=real(), omega=real())
def test_torque_adjust(inertia_rot, tau_rot, phi_0, phi, omega_0, omega):
    if tau_rot == 0.0:
        with pytest.raises(ZeroDivisionError):
            torque_adjust(inertia_rot, tau_rot, phi_0, phi, omega_0, omega)
    else:
        ans = torque_adjust(inertia_rot, tau_rot, phi_0, phi, omega_0,
                            omega)
        assert isinstance(ans, float)


@given(h=real(), n=vector(), a=positive(), b=positive())
def test_force_social_helbing(h, n, a, b):
    if b == 0.0:
        with pytest.raises(ZeroDivisionError):
            force_social_helbing(h, n, a, b)
    else:
        ans = force_social_helbing(h, n, a, b)
        assert isinstance(ans, np.ndarray)


@given(h=real(), n=vector(), v=vector(), t=vector(), mu=positive(),
       kappa=positive(), damping=positive())
def test_force_contact(h, n, v, t, mu, kappa, damping):
    ans = force_contact(h, n, v, t, mu, kappa, damping)
    assert isinstance(ans, np.ndarray)


@given(x0=vector(), r0=positive(), x1=vector(), r1=positive())
def test_distance_circle_circle(x0, r0, x1, r1):
    h, n = distance_circle_circle(x0, r0, x1, r1)

    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)

    x = x0 - x1
    r_tot = r0 + r1

    assert h >= -r_tot

    if np.all(x == 0.0):
        assert np.isclose(length(n), 0.0)
    elif np.allclose(x, 0.0):
        # Very small floats cause trouble
        # Don't know if it can be easily fixed
        pass
    else:
        assert np.allclose(length(n), 1.0)


@given(x0=three_vectors(), r0=three_positive(),
       x1=three_vectors(), r1=three_positive(), )
def test_distance_three_circle(x0, r0, x1, r1):
    h, n, r_moment0, r_moment1 = distance_three_circle(x0, r0, x1, r1)

    assert isinstance(h, float)
    assert isinstance(n, np.ndarray)
    assert isinstance(r_moment0, np.ndarray)
    assert isinstance(r_moment1, np.ndarray)


@given(x=vector(), r=positive(), p=line(), )
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


@given(points=vectors(elements=st.floats(0.0, 1.0)),
       cell_width=st.floats(0.1, 1.0))
def test_block_list(points, cell_width):
    index_list, count, offset, x_min, x_max = block_list(points, cell_width)

    assert isinstance(index_list, np.ndarray)
    assert index_list.dtype.type is np.int64

    assert isinstance(count, np.ndarray)
    assert count.dtype.type is np.int64

    assert isinstance(offset, np.ndarray)
    assert offset.dtype.type is np.int64

    assert isinstance(x_min, np.ndarray)
    assert x_min.dtype.type is np.int64

    assert isinstance(x_max, np.ndarray)
    assert x_max.dtype.type is np.int64
