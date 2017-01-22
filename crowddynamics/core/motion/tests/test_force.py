import numpy as np
import pytest
from hypothesis import given

from crowddynamics.core.motion import force_fluctuation, force_adjust, \
    force_social_helbing, force_contact
from crowddynamics.testing import vector, real


@given(mass=real(min_value=0), scale=real(min_value=0))
def test_force_fluctuation(mass, scale):
    ans = force_fluctuation(mass, scale)
    assert isinstance(ans, np.ndarray)


@given(mass=real(min_value=0), tau_adj=real(min_value=0), v0=real(min_value=0),
       e0=vector(), v=vector())
def test_force_adjust(mass, tau_adj, v0, e0, v):
    if tau_adj == 0.0:
        with pytest.raises(ZeroDivisionError):
            force_adjust(mass, tau_adj, v0, e0, v)
    else:
        ans = force_adjust(mass, tau_adj, v0, e0, v)
        assert isinstance(ans, np.ndarray)


@given(h=real(), n=vector(), a=real(min_value=0), b=real(min_value=0))
def test_force_social_helbing(h, n, a, b):
    if b == 0.0:
        with pytest.raises(ZeroDivisionError):
            force_social_helbing(h, n, a, b)
    else:
        ans = force_social_helbing(h, n, a, b)
        assert isinstance(ans, np.ndarray)


@given(h=real(), n=vector(), v=vector(), t=vector(), mu=real(min_value=0),
       kappa=real(min_value=0), damping=real(min_value=0))
def test_force_contact(h, n, v, t, mu, kappa, damping):
    ans = force_contact(h, n, v, t, mu, kappa, damping)
    assert isinstance(ans, np.ndarray)
