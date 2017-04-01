import numpy as np
import pytest
from hypothesis import given

from crowddynamics.core.motion import force_fluctuation, force_adjust, \
    force_social_helbing, force_contact
from crowddynamics.testing import real


SIZE = 10


@given(
    mass=real(min_value=0),
    scale=real(min_value=0)
)
def test_force_fluctuation(mass, scale):
    ans = force_fluctuation(mass, scale)
    assert isinstance(ans, np.ndarray)
    assert ans.dtype.type is np.float64
    assert ans.shape == (1, 2)


@given(
    mass=real(min_value=0, shape=(SIZE, 1)),
    scale=real(min_value=0, shape=SIZE)
)
def test_force_fluctuation_vector(mass, scale):
    ans = force_fluctuation(mass, scale)
    assert isinstance(ans, np.ndarray)
    assert ans.dtype.type is np.float64
    assert ans.shape == (SIZE, 2)


@given(
    mass=real(min_value=0),
    tau_adj=real(min_value=0, exclude_zero='near'),
    v0=real(min_value=0),
    e0=real(shape=2),
    v=real(shape=2)
)
def test_force_adjust(mass, tau_adj, v0, e0, v):
    ans = force_adjust(mass, tau_adj, v0, e0, v)
    assert isinstance(ans, np.ndarray)
    assert ans.dtype.type is np.float64
    assert ans.shape == (2,)


@given(
    mass=real(min_value=0, shape=(SIZE, 1)),
    tau_adj=real(min_value=0, exclude_zero='near', shape=(SIZE, 1)),
    v0=real(min_value=0, shape=(SIZE, 1)),
    e0=real(shape=(SIZE, 2)),
    v=real(shape=(SIZE, 2))
)
def test_force_adjust_vector(mass, tau_adj, v0, e0, v):
    ans = force_adjust(mass, tau_adj, v0, e0, v)
    assert isinstance(ans, np.ndarray)
    assert ans.dtype.type is np.float64
    assert ans.shape == (SIZE, 2)


@given(
    h=real(),
    n=real(shape=2),
    a=real(min_value=0),
    b=real(min_value=0, exclude_zero='near')
)
def test_force_social_helbing(h, n, a, b):
    ans = force_social_helbing(h, n, a, b)
    assert isinstance(ans, np.ndarray)
    assert ans.dtype.type is np.float64
    assert ans.shape == (2,)


@given(
    h=real(),
    n=real(shape=2),
    v=real(shape=2),
    t=real(shape=2),
    mu=real(min_value=0),
    kappa=real(min_value=0),
    damping=real(min_value=0)
)
def test_force_contact(h, n, v, t, mu, kappa, damping):
    ans = force_contact(h, n, v, t, mu, kappa, damping)
    assert isinstance(ans, np.ndarray)
    assert ans.dtype.type is np.float64
    assert ans.shape == (2,)
