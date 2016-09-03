import numpy as np
import numba
from numba import f8, void

"""
Functions operating on 2-Dimensional vectors.
"""
# TODO: Normal, Tangent


@numba.vectorize([f8(f8)])
def wrap_to_pi(rad):
    """
    Wraps angles in rad in radians, to the interval [−pi pi].

    Pi maps to pi and −pi maps to −pi. (In general, odd, positive multiples of
    pi map to pi and odd, negative multiples of pi map to −pi.)

    [Matlab](http://se.mathworks.com/help/map/ref/wraptopi.html)

    :param rad: Angle in radians.
    :return: Angle in [-pi, pi].
    """
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
    """90 degree counterclockwise rotation for 2D vector."""
    rot = np.zeros_like(vec2d)
    rot[0] = -vec2d[1]
    rot[1] = vec2d[0]
    return rot


@numba.jit(f8[:](f8[:]), nopython=True, nogil=True)
def rotate270(vec2d):
    """90 degree clockwise rotation for 2D vector."""
    rot = np.zeros_like(vec2d)
    rot[0] = vec2d[1]
    rot[1] = -vec2d[0]
    return rot


@numba.jit(f8(f8[:]), nopython=True, nogil=True)
def angle(vec2d):
    """Angle of 2d vector in radians.
    :param vec2d: 2D vector
    :return: Angle in [-pi, pi]
    """
    return np.arctan2(vec2d[1], vec2d[0])


@numba.jit(f8[:](f8[:, :]), nopython=True, nogil=True)
def angle_nx2(vec2d):
    """Angle of 2d vectors in radians."""
    return np.arctan2(vec2d[:, 1], vec2d[:, 0])


@numba.jit(f8(f8[:]), nopython=True, nogil=True)
def length(vec2d):
    """
    :param vec2d: 2D vector
    :return: Length of the vector in [0, infty)
    """
    return np.hypot(vec2d[0], vec2d[1])


@numba.jit(f8[:](f8[:, :]), nopython=True, nogil=True)
def length_nx2(vec2d):
    return np.hypot(vec2d[:, 0], vec2d[:, 1])


@numba.jit(f8(f8[:], f8[:]), nopython=True, nogil=True)
def dot2d(v0, v1):
    """Dot product for 2D vectors."""
    return v0[0] * v1[0] + v0[1] * v1[1]


@numba.jit(f8(f8[:], f8[:]), nopython=True, nogil=True)
def cross2d(v0, v1):
    """Cross product for 2D vectors. Right corner from 3D cross product."""
    return v0[0] * v1[1] - v0[1] * v1[0]


@numba.jit(f8[:](f8[:]), nopython=True, nogil=True)
def normalize(vec2d):
    return vec2d / np.hypot(vec2d[0], vec2d[1])


@numba.jit(f8[:, :](f8[:, :]), nopython=True, nogil=True)
def normalize_nx2(vec2d):
    return vec2d / np.hypot(vec2d[:, 0], vec2d[:, 1]).reshape((len(vec2d), 1))


@numba.jit(void(f8[:], f8), nopython=True, nogil=True)
def truncate(vec2d, limit):
    l = length(vec2d)
    if l != 0 and l > limit:
        vec2d *= limit / l


@numba.jit(nopython=True, nogil=True)
def distance_circle_line(x, r, p):
    d = p[1] - p[0]
    l_w = length(d)
    t_w = d / l_w
    n_w = rotate90(t_w)

    q = x - p
    l_t = - dot2d(t_w, q[1]) - dot2d(t_w, q[0])

    if l_t > l_w:
        d_iw = length(q[0])
    elif l_t < -l_w:
        d_iw = length(q[1])
    else:
        l_n = dot2d(n_w, q[0])
        d_iw = np.abs(l_n)

    d_iw -= r
    return d_iw


@numba.jit(nopython=True, nogil=True)
def distance_circle_line_with_normal(x, r, p):
    d = p[1] - p[0]
    l_w = length(d)
    t_w = d / l_w
    n_w = rotate90(t_w)

    q = x - p
    l_t = - dot2d(t_w, q[1]) - dot2d(t_w, q[0])

    if l_t > l_w:
        d_iw = length(q[0])
        n_iw = q[0] / d_iw
    elif l_t < -l_w:
        d_iw = length(q[1])
        n_iw = q[1] / d_iw
    else:
        l_n = dot2d(n_w, q[0])
        d_iw = np.abs(l_n)
        n_iw = np.sign(l_n) * n_w

    d_iw -= r
    return d_iw, n_iw
