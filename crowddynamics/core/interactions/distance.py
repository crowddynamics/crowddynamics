r"""
Distance functions for potentials
"""

import numba
import numpy as np

from crowddynamics.core.vector2D import length, rotate90, dot2d


@numba.jit(nopython=True, nogil=True)
def distance_circle_circle(x0, r0, x1, r1):
    r"""
    Skin-to-Skin distance with normal

    Args:
        x0 (numpy.ndarray):
        r0 (float):
        x1 (numpy.ndarray):
        r1 (float):

    Returns:
        (float, numpy.ndarray): (skin-to-skin distance, normal vector)
    """
    x = x0 - x1
    d = length(x)
    r_tot = r0 + r1
    h = d - r_tot
    if d == 0.0:
        n = np.zeros(2)
    else:
        n = x / d

    return h, n


@numba.jit(nopython=True, nogil=True)
def distance_three_circle(x0, r0, x1, r1):
    r"""
    Skin-to-Skin distance between two three-circle models.

    Args:
        x0 ((numpy.ndarray, numpy.ndarray, numpy.ndarray)):
        r0 ((float, float, float)):
        x1 ((numpy.ndarray, numpy.ndarray, numpy.ndarray)):
        r1 ((float, float, float)):

    Returns:
        (float, numpy.ndarray, numpy.ndarray, numpy.ndarray):
    """
    h_min = np.nan
    normal = np.zeros(2)
    i_min = 0
    j_min = 0

    for i, (xi, ri) in enumerate(zip(x0, r0)):
        for j, (xj, rj) in enumerate(zip(x1, r1)):
            h, n = distance_circle_circle(xi, ri, xj, rj)
            if h < h_min or np.isnan(h_min):
                h_min = h
                normal = n
                i_min = i
                j_min = j

    r_moment0 = x0[i_min] + r0[i_min] * normal - x0[0]
    r_moment1 = x0[j_min] - r1[j_min] * normal - x1[0]

    return h_min, normal, r_moment0, r_moment1


@numba.jit(nopython=True, nogil=True)
def distance_circle_line(x, r, p):
    r"""
    Skin-to-Skin distance between circle and line

    Args:
        x (numpy.ndarray):
        r (float):
        p (numpy.ndarray):

    Returns:
        (float, numpy.ndarray): (skin-to-skin distance, normal vector)
    """
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

    h_iw = d_iw - r

    return h_iw, n_iw


@numba.jit(nopython=True, nogil=True)
def distance_three_circle_line(x, r, p):
    r"""
    Skin-to-Skin distance between three circle model and line

    Args:
        x ((numpy.ndarray, numpy.ndarray, numpy.ndarray)):
        r ((float, float, float)):
        p (numpy.ndarray):

    Returns:
        (float, numpy.ndarray, numpy.ndarray)
    """
    h_min = np.nan
    normal = np.zeros(2)
    i_min = 0

    for i, (x_, r_) in enumerate(zip(x, r)):
        h, n = distance_circle_line(x_, r_, p)
        if h < h_min or np.isnan(h_min):
            h_min = h
            normal = n
            i_min = i

    r_moment = x[i_min] - r[i_min] * normal - x[0]

    return h_min, normal, r_moment


@numba.jit(nopython=True, nogil=True)
def overlapping_circle_circle(x, r, start_index, i):
    """
    Test if two circles are overlapping.

    Args:
        x:
        r:
        start_index:
        i:

    Returns:
        Boolean:

    """
    for j in range(start_index, i):
        h, _ = distance_circle_circle(x[i], x[j], r[i], r[j])
        if h < 0:
            return True
    return False


@numba.jit(nopython=True, nogil=True)
def overlapping_three_circle(x, r, start_index, i):
    """
    Test if two three-circle models are overlapping.

    Args:
        x (collections.namedtuple):
            Attributes:
            - position=numpy.ndarray
            - position_ls=numpy.ndarray
            - position_rs=numpy.ndarray

        r (collections.namedtuple):
            Attributes:
            - r_t
            - r_s

        start_index:
        i:

    Returns:
        Boolean:

    """
    for j in range(start_index, i):
        h, _, _, _ = distance_three_circle(
            (x.position[i], x.position_ls[i], x.position_rs[i]),
            (r.r_t[i], r.r_s[i], r.r_s[i]),
            (x.position[j], x.position_ls[j], x.position_rs[j]),
            (r.r_t[j], r.r_s[j], r.r_s[j])
        )
        if h < 0:
            return True
    return False
