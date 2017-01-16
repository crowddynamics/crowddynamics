import numpy as np
import pytest
from hypothesis import given

from crowddynamics.core.motion import force_fluctuation, torque_fluctuation, \
    force_adjust, torque_adjust, force_social_helbing, force_contact
from crowddynamics.tests.strategies import positive, vector, real


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
