import numpy as np
import numba
from numba import f8, void


"""
Functions operating on 2-Dimensional vectors.
"""


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
def force_limit(force, f_max):
    f_mag = np.hypot(force[0], force[1])
    if f_mag > f_max:
        force *= f_max / f_mag


@numba.vectorize([f8(f8)])
def wrap_to_pi(angle):
    angle %= 2 * np.pi
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle
