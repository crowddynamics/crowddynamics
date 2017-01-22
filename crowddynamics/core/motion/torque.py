import numba
import numpy as np
from scipy.stats import truncnorm

from crowddynamics.core.vector2D import wrap_to_pi


def torque_fluctuation(inertia_rot, scale):
    r"""
    Random torque.

    .. math::
       \eta \sim \mathcal{N}(\mu, \sigma^{2})

    Args:
        inertia_rot (float):
            Rotational intertial

        scale (float):
            Standard deviation of truncated normal distribution

    Returns:
        numpy.ndarray: Fluctuation torque
    """
    if isinstance(inertia_rot, np.ndarray):
        size = len(inertia_rot)
    elif isinstance(scale, np.ndarray):
        size = len(scale)
    else:
        size = 1

    torque = truncnorm.rvs(-3.0, 3.0, loc=0.0, scale=scale, size=size)
    return torque * inertia_rot


@numba.jit(nopython=True, nogil=True)
def torque_adjust(inertia_rot, tau_rot, phi_0, phi, omega_0, omega):
    r"""
    Adjusting torque account for agent's desire to rotate it orientation.

    .. math::
       M_{adj} = \frac{I_{rot}}{\tau_{rot}} \left( \omega_{0} \left (
                 \frac{\varphi - \varphi_{0}}{\pi} \right ) - \omega\right),

    Angular difference :math:`\varphi - \varphi_{0}` is wrapped between interval
    :math:`[-\pi, \pi]` so that division by :math:`\pi` returns value between
    :math:`[-1, 1]`. This gives direction and magnitude for the torque.

    Args:
        inertia_rot (float):
            Rotational inertia

        tau_rot (float):
            Characteristic time :math:`\tau_{rot}` time for agent to adjust it
            orientation.

        phi_0 (float):
            Target orientation :math:`\varphi_{0}`. In low and medium crowd
            densities the angle of the target direction can be sufficient for
            target orientation. In high crowd densities agents may twist their
            body differently for example to try to squeeze through narrow
            spaces, requiring more sophisticated algorithms.

        phi (float):
            Current orientation :math:`\varphi`

        omega_0 (float):
            Maximum angular velocity :math:`\omega_{0}`.

        omega (float):

    Returns:
        float:
    """
    return inertia_rot / tau_rot * \
           (wrap_to_pi(phi_0 - phi) / np.pi * omega_0 - omega)
