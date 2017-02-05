import numpy as np
from hypothesis import given

from crowddynamics.core.motion import torque_fluctuation, torque_adjust
from crowddynamics.testing import real

SIZE = 10


@given(inertia_rot=real(0), scale=real(0))
def test_torque_fluctuation(inertia_rot, scale):
    ans = torque_fluctuation(inertia_rot, scale)
    assert isinstance(ans, np.ndarray)
    assert ans.dtype.type is np.float64
    assert ans.shape == (1,)


@given(inertia_rot=real(0, shape=SIZE), scale=real(0, shape=SIZE))
def test_torque_fluctuation_vector(inertia_rot, scale):
    ans = torque_fluctuation(inertia_rot, scale)
    assert isinstance(ans, np.ndarray)
    assert ans.dtype.type is np.float64
    assert ans.shape == (SIZE,)


@given(inertia_rot=real(0),
       tau_rot=real(0, exclude_zero='near'),
       phi_0=real(),
       phi=real(),
       omega_0=real(),
       omega=real())
def test_torque_adjust(inertia_rot, tau_rot, phi_0, phi, omega_0, omega):
    ans = torque_adjust(inertia_rot, tau_rot, phi_0, phi, omega_0,
                        omega)
    assert isinstance(ans, float)


@given(inertia_rot=real(0, shape=SIZE),
       tau_rot=real(0, exclude_zero='near', shape=SIZE),
       phi_0=real(shape=SIZE),
       phi=real(shape=SIZE),
       omega_0=real(shape=SIZE),
       omega=real(shape=SIZE))
def test_torque_adjust_vector(inertia_rot, tau_rot, phi_0, phi, omega_0, omega):
    ans = torque_adjust(inertia_rot, tau_rot, phi_0, phi, omega_0,
                        omega)
    assert isinstance(ans, np.ndarray)
    assert ans.dtype.type is np.float64
    assert ans.shape == (SIZE,)
