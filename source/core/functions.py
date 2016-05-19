import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def rotate90(vec2d):
    """
    90 degree counterclockwise rotation for 2D vector.

    (x, y) -> (-y, x)

    https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
    """
    rot = np.zeros_like(vec2d)
    rot[0] = -vec2d[1]
    rot[1] = vec2d[0]
    return rot


@numba.jit(nopython=True, nogil=True)
def rotate270(vec2d):
    """
    90 degree clockwise rotation for 2D vector.

    (x, y) -> (y, -x)

    https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
    """
    rot = np.zeros_like(vec2d)
    rot[0] = vec2d[1]
    rot[1] = -vec2d[0]
    return rot

