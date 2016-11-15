r"""
Distance functions for potentials
"""

import numba
import numpy as np

from crowddynamics.core.vector2D import length, rotate90, dot2d


@numba.jit(nopython=True, nogil=True)
def distance_circle_circle(x0, r0, x1, r1):
    """
    Skin-to-Skin distance with normal

    Args:
        x0 (numpy.ndarray):
        r0 (float):
        x1 (numpy.ndarray):
        r1 (float):

    Returns:
        (float, float): (skin-to-skin distance, normal vector)
    """
    x = x0 - x1
    d = length(x)
    r_tot = r0 + r1
    h = d - r_tot
    n = x / d

    return h, n


@numba.jit(nopython=True, nogil=True)
def distance_three_circle(x0, r0, x1, r1):
    """
    Distance between two three-circle models.

    Args:
        x0 ((numpy.ndarray, numpy.ndarray, numpy.ndarray)):
        r0 ((float, float, float)):
        x1 ((numpy.ndarray, numpy.ndarray, numpy.ndarray)):
        r1 ((float, float, float)):

    Returns:
        (float, float, numpy.ndarray, numpy.ndarray):
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
    """

    Args:
        x (numpy.ndarray):
        r (float:
        p (numpy.ndarray):

    Returns:
        (float, float): (skin-to-skin distance, normal vector)
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

    d_iw -= r

    return d_iw, n_iw


@numba.jit(nopython=True, nogil=True)
def distance_three_circle_line(x, r, p):
    """

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
        d, n = distance_circle_line(x_, r_, p)
        h = d - r_
        if h < h_min or np.isnan(h_min):
            h_min = h
            normal = n
            i_min = i

    r_moment = x[i_min] - r[i_min] * normal - x[0]

    return h_min, normal, r_moment
