import numpy as np
from hypothesis import given

from crowddynamics.core.motion.adjusting import force_adjust, torque_adjust
from crowddynamics.core.motion.contact import force_contact
from crowddynamics.core.motion.fluctuation import force_fluctuation, \
    torque_fluctuation
from crowddynamics.core.motion.helbing import \
    force_social_helbing
from crowddynamics.testing import reals

SIZE = 10


@given(mass=reals(min_value=0, shape=SIZE),
       scale=reals(min_value=0, shape=SIZE))
def test_force_fluctuation(mass, scale):
    ans = force_fluctuation(mass, scale)
    assert isinstance(ans, np.ndarray)
    assert ans.dtype.type is np.float64
    assert ans.shape == (SIZE, 2)


@given(mass=reals(min_value=0),
       tau_adj=reals(min_value=0, exclude_zero='near'),
       v0=reals(min_value=0),
       e0=reals(shape=2),
       v=reals(shape=2))
def test_force_adjust(mass, tau_adj, v0, e0, v):
    ans = force_adjust(mass, tau_adj, v0, e0, v)
    assert isinstance(ans, np.ndarray)
    assert ans.dtype.type is np.float64
    assert ans.shape == (2,)


@given(h=reals(),
       n=reals(shape=2),
       a=reals(min_value=0),
       b=reals(min_value=0, exclude_zero='near'))
def test_force_social_helbing(h, n, a, b):
    ans = force_social_helbing(h, n, a, b)
    assert isinstance(ans, np.ndarray)
    assert ans.dtype.type is np.float64
    assert ans.shape == (2,)


@given(h=reals(),
       n=reals(shape=2),
       v=reals(shape=2),
       t=reals(shape=2),
       mu=reals(min_value=0),
       kappa=reals(min_value=0),
       damping=reals(min_value=0))
def test_force_contact(h, n, v, t, mu, kappa, damping):
    ans = force_contact(h, n, v, t, mu, kappa, damping)
    assert isinstance(ans, np.ndarray)
    assert ans.dtype.type is np.float64
    assert ans.shape == (2,)


@given(inertia_rot=reals(0, shape=SIZE), scale=reals(0, shape=SIZE))
def test_torque_fluctuation(inertia_rot, scale):
    ans = torque_fluctuation(inertia_rot, scale)
    assert isinstance(ans, np.ndarray)
    assert ans.dtype.type is np.float64
    assert ans.shape == (SIZE,)


@given(inertia_rot=reals(0),
       tau_rot=reals(0, exclude_zero='near'),
       phi_0=reals(),
       phi=reals(),
       omega_0=reals(),
       omega=reals())
def test_torque_adjust(inertia_rot, tau_rot, phi_0, phi, omega_0, omega):
    ans = torque_adjust(inertia_rot, tau_rot, phi_0, phi, omega_0,
                        omega)
    assert isinstance(ans, float)
