import numba
import numpy as np

from .vector2d import dot2d, truncate


@numba.jit(nopython=True, nogil=True)
def force_social2(x_rel, v_rel, r_tot, r_minor, mass, k, tau_0, f_max):
    """Modified for three circle model."""
    force = np.zeros_like(x_rel)

    a = dot2d(v_rel, v_rel)
    b = -dot2d(x_rel, v_rel)
    c = dot2d(x_rel, x_rel) - r_tot ** 2
    c2 = dot2d(x_rel, x_rel) - r_minor ** 2
    d = np.sqrt(b ** 2 - a * c)
    d2 = np.sqrt(b ** 2 - a * c2)

    # Avoid zero division.
    # No interaction if tau cannot be defined.
    if np.isnan(d) or d == 0 or a == 0:
        return force

    tau = (b - d) / a  # Time-to-collision. In seconds
    tau_max = 30.0     # Maximum time for interaction.

    if tau <= 0 or tau > tau_max:
        return force

    tau2 = (b - d2) / a  # Time-to-collision. In seconds

    # Force is returned negative as repulsive force
    coeff = - mass * k
    magitude = 1 / (a * tau ** 2.0) * np.exp(-tau / tau_0) * (2.0 / tau + 1.0 / tau_0)
    magitude2 = 1 / (a * tau2 ** 2.0) * np.exp(-tau2 / tau_0) * (2.0 / tau2 + 1.0 / tau_0)
    direction = v_rel - (v_rel * b + x_rel * a) / d

    force += coeff * magitude * direction

    # Truncation for small tau
    truncate(force, f_max)

    return force
