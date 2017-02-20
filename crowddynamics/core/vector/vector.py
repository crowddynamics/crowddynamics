r"""Functions operating on 2-Dimensional vectors."""
import numba
import numpy as np
from numba import f8, void


vector_type = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
])


@numba.vectorize([f8(f8)])
def wrap_to_pi(rad):
    r"""
    Wraps angles in rad in radians, to the interval [−pi pi].

    .. math::
       \varphi : \mathbb{R} \to [−pi pi]

    Pi maps to pi and −pi maps to −pi. (In general, odd, positive multiples of
    pi map to pi and odd, negative multiples of pi map to −pi.) [wraptopi]_

    Args:
        rad (float): Angle in radians

    Returns:
        float: Angle within [-pi, pi]

    References:

    .. [wraptopi] http://se.mathworks.com/help/map/ref/wraptopi.html
    """
    # TODO: simplify
    rad_ = rad % (2 * np.pi)
    if rad < 0 and rad_ == np.pi:
        # negative multiples of pi map to −pi
        return -np.pi
    elif rad_ > np.pi:
        return rad_ - (2 * np.pi)
    else:
        return rad_


def unit_vector(orientation):
    return np.array([np.cos(orientation), np.sin(orientation)])
