import numpy as np
import numba


@numba.jit(nopython=True, nogil=True)
def torque_random(agent):
    """Random torque."""
    for i in range(agent.size):
        return np.random.uniform(-1, 1)


@numba.jit(nopython=True, nogil=True)
def torque_adjust(moment_rot, phi, omega, phi_0, omega_0, tau):
    """Adjusting torque."""
    return moment_rot / tau * ((phi - phi_0) * omega_0 - omega)


@numba.jit(nopython=True, nogil=True)
def torque(radius, force):
    """Torque for 2D vectors. Right corner from cross product."""
    return radius[0] * force[1] - radius[1] * force[0]
