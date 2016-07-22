import numba
import numpy as np

from .vector2d import dot2d, truncate


def time_to_collision(x_rel, v_rel, r_tot):
    """Time-to-collision for circle."""
    a = dot2d(v_rel, v_rel)
    b = -dot2d(x_rel, v_rel)
    c = dot2d(x_rel, x_rel) - r_tot ** 2
    d = np.sqrt(b ** 2 - a * c)

    # Avoid zero division.
    # No interaction if tau cannot be defined.
    if np.isnan(d) or d == 0 or a == 0:
        return np.nan

    tau = (b - d) / a  # Time-to-collision. In seconds
    return tau


@numba.jit(nopython=True, nogil=True)
def force_social2(x_rel, v_rel, r_tot, mass, k, tau_0, f_max):
    """Modified for three circle model."""
    force = np.zeros_like(x_rel)

    a = dot2d(v_rel, v_rel)
    b = -dot2d(x_rel, v_rel)
    c = dot2d(x_rel, x_rel) - r_tot ** 2
    d = np.sqrt(b ** 2 - a * c)

    # Avoid zero division.
    # No interaction if tau cannot be defined.
    if np.isnan(d) or d == 0 or a == 0:
        return force

    tau = (b - d) / a  # Time-to-collision. In seconds
    tau_max = 30.0     # Maximum time for interaction.

    if tau <= 0 or tau > tau_max:
        return force

    # Force is returned negative as repulsive force
    coeff = - mass * k
    magitude = 1 / (a * tau ** 2.0) * np.exp(-tau / tau_0) * (2.0 / tau + 1.0 / tau_0)
    direction = v_rel - (v_rel * b + x_rel * a) / d

    force += coeff * magitude * direction

    # Truncation for small tau
    truncate(force, f_max)

    return force
