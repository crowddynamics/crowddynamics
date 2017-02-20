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


@numba.jit(f8[:](f8[:]), nopython=True, nogil=True)
def rotate90(vec2d):
    r"""
    90 degree counterclockwise rotation for 2D vector.

    Args:
        vec2d (numpy.ndarray):

    Returns:
        numpy.ndarray:
    """
    rot = np.zeros_like(vec2d)
    rot[0] = -vec2d[1]
    rot[1] = vec2d[0]
    return rot


@numba.jit(f8[:](f8[:]), nopython=True, nogil=True)
def rotate270(vec2d):
    r"""
    90 degree clockwise rotation for 2D vector.

    Args:
        vec2d (numpy.ndarray):

    Returns:
        numpy.ndarray:
    """
    rot = np.zeros_like(vec2d)
    rot[0] = vec2d[1]
    rot[1] = -vec2d[0]
    return rot


@numba.jit(f8(f8[:]), nopython=True, nogil=True)
def angle(vec2d):
    r"""
    Angle of 2d vector in radians.

    Args:
        vec2d (numpy.ndarray): 2D vector

    Returns:
        float: Angle in [-pi, pi]
    """
    return np.arctan2(vec2d[1], vec2d[0])


# TODO: unify with angle
@numba.jit(f8[:](f8[:, :]), nopython=True, nogil=True)
def angle_nx2(vec2d):
    r"""
    Angle of 2d vectors in radians.

    Args:
        vec2d (numpy.ndarray):

    Returns:
        numpy.ndarray:
    """
    return np.arctan2(vec2d[:, 1], vec2d[:, 0])


@numba.jit(f8(f8[:]), nopython=True, nogil=True)
def length(vec2d):
    r"""
    Length

    Args:
        vec2d (numpy.ndarray): 2D vector

    Returns:
        float: Length of the vector in [0, infty)
    """
    return np.hypot(vec2d[0], vec2d[1])


# TODO: unifyg with length
@numba.jit(f8[:](f8[:, :]), nopython=True, nogil=True)
def length_nx2(vec2d):
    r"""
    Length

    Args:
        vec2d (numpy.ndarray):

    Returns:
        numpy.ndarray:
    """
    return np.hypot(vec2d[:, 0], vec2d[:, 1])


@numba.jit(f8(f8[:], f8[:]), nopython=True, nogil=True)
def dot(v0, v1):
    r"""
    Dot product for 2D vectors.

    Args:
        v0 (numpy.ndarray):
        v1 (numpy.ndarray):

    Returns:
        float:
    """
    return v0[0] * v1[0] + v0[1] * v1[1]


@numba.jit(f8(f8[:], f8[:]), nopython=True, nogil=True)
def cross(v0, v1):
    r"""
    Cross product for 2D vectors. Right corner from 3D cross product.

    Args:
        v0 (numpy.ndarray):
        v1 (numpy.ndarray):

    Returns:
        float:
    """
    return v0[0] * v1[1] - v0[1] * v1[0]


@numba.jit(f8[:](f8[:]), nopython=True, nogil=True)
def normalize(vec2d):
    r"""
    Normalize

    Args:
        vec2d (numpy.ndarray):

    Returns:
        numpy.ndarray:
    """
    if np.all(vec2d == 0.0):
        return np.zeros(2)
    return vec2d / length(vec2d)


@numba.jit(f8[:, :](f8[:, :]), nopython=True, nogil=True)
def normalize_nx2(vec2d):
    r"""
    Normalize

    Args:
        vec2d (numpy.ndarray):

    Returns:
        numpy.ndarray:
    """
    return vec2d / np.hypot(vec2d[:, 0], vec2d[:, 1]).reshape((len(vec2d), 1))


@numba.jit(void(f8[:], f8), nopython=True, nogil=True)
def truncate(v, l):
    r"""
    Truncate vector :math:`\mathbf{v}` to length :math:`l > 0` if
    :math:`\|\mathbf{v}\| > l`.
    
    .. math::
        \begin{cases}
        l \frac{\mathbf{v}}{\|\mathbf{v}\|} & \|\mathbf{v}\| > l \\
        \mathbf{v} & \|\mathbf{v}\| \leq l \\
        \end{cases}

    Args:
        v (numpy.ndarray):
        l (float):

    Returns:
        None:
    """
    vlen = length(v)
    if vlen > l:
        v *= l / vlen
