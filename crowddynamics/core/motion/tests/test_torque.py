import numpy as np
import pytest
from hypothesis import given

from crowddynamics.core.motion import torque_fluctuation, torque_adjust
from crowddynamics.testing import real


@given(inertia_rot=real(min_value=0), scale=real(min_value=0))
def test_torque_fluctuation(inertia_rot, scale):
    ans = torque_fluctuation(inertia_rot, scale)
    assert isinstance(ans, np.ndarray)


@given(inertia_rot=real(min_value=0), tau_rot=real(min_value=0), phi_0=real(), phi=real(),
       omega_0=real(), omega=real())
def test_torque_adjust(inertia_rot, tau_rot, phi_0, phi, omega_0, omega):
    if tau_rot == 0.0:
        with pytest.raises(ZeroDivisionError):
            torque_adjust(inertia_rot, tau_rot, phi_0, phi, omega_0, omega)
    else:
        ans = torque_adjust(inertia_rot, tau_rot, phi_0, phi, omega_0,
                            omega)
        assert isinstance(ans, float)
