r"""Functions operating on 2-Dimensional vectors."""
import numba
import numpy as np
from numba import f8, void
from numba.types import Float, Array


@numba.vectorize([f8(f8)], cache=True)
def wrap_to_pi(rad):
    r"""Wraps angles in rad in radians, to the interval [−pi pi].

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


@numba.jit(f8[:](f8[:]), nopython=True, nogil=True, cache=True)
def rotate90(v):
    r"""90 degree counterclockwise rotation for 2D vector.

    .. tikz::
       \begin{scope}[scale=0.5]
       \draw[color=gray!20] (-1, -1) grid (6, 6);
       \draw[thick, gray, dashed, ->] (0, 0) -- (90:5);
       \draw[thick, ->] (0, 0) -- (0:5);
       \end{scope}

    Args:
        v (numpy.ndarray):

    Returns:
        numpy.ndarray:
    """
    rot = np.zeros_like(v)
    rot[0] = -v[1]
    rot[1] = v[0]
    return rot


@numba.jit(f8[:](f8[:]), nopython=True, nogil=True, cache=True)
def rotate270(v):
    r"""90 degree clockwise rotation for 2D vector.

    .. tikz::
       \begin{scope}[scale=0.5]
       \draw[color=gray!20] (-1, -1) grid (6, 6);
       \draw[thick, gray, dashed, ->] (0, 0) -- (0:5);
       \draw[thick, ->] (0, 0) -- (90:5);
       \end{scope}

    Args:
        v (numpy.ndarray):

    Returns:
        numpy.ndarray:
    """
    rot = np.zeros_like(v)
    rot[0] = v[1]
    rot[1] = -v[0]
    return rot


@numba.jit([f8(f8[:]), f8[:](f8[:, :])],
           nopython=True, nogil=True, cache=True)
def angle(v):
    r"""Angle of 2d vector in radians (angle between vector and x-axis).

    Args:
        v (numpy.ndarray): 2D vector

    Returns:
        float|numpy.ndarray: Angle in [-pi, pi]
    """
    return np.arctan2(v[..., 1], v[..., 0])


@numba.jit([f8(f8[:]), f8[:](f8[:, :])],
           nopython=True, nogil=True, cache=True)
def length(v):
    r"""Length of an vector

    .. math::
       \|\mathbf{v}\|

    Args:
        v (numpy.ndarray): 2D vector

    Returns:
        float|numpy.ndarray: Length of the vector in [0, infty)
    """
    return np.hypot(v[..., 0], v[..., 1])


@numba.jit([f8(f8[:], f8[:]), f8[:](f8[:, :], f8[:, :])],
           nopython=True, nogil=True, cache=True)
def dot(v0, v1):
    r"""Dot product for 2D vectors.

    .. math::
       \mathbf{v}_0 \cdot \mathbf{v}_1

    Args:
        v0 (numpy.ndarray):
        v1 (numpy.ndarray):

    Returns:
        float:
    """
    return v0[..., 0] * v1[..., 0] + v0[..., 1] * v1[..., 1]


@numba.jit([f8(f8[:], f8[:]), f8[:](f8[:, :], f8[:, :])],
           nopython=True, nogil=True, cache=True)
def cross(v0, v1):
    r"""Cross product for 2D vectors. Right corner from 3D cross product.

    .. math::
       \mathbf{v}_0 \times \mathbf{v}_1

    Args:
        v0 (numpy.ndarray):
        v1 (numpy.ndarray):

    Returns:
        float:
    """
    return v0[..., 0] * v1[..., 1] - v0[..., 1] * v1[..., 0]


@numba.jit(f8[:](f8[:]), nopython=True, nogil=True, cache=True)
def normalize(v):
    r"""Normalize

    Args:
        v (numpy.ndarray):

    Returns:
        numpy.ndarray:
    """
    l = length(v)
    return v / l if l else v


@numba.jit(void(f8[:], f8), nopython=True, nogil=True, cache=True)
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


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def unit_vector(orientation):
    """Unit vector"""
    if isinstance(orientation, Float):
        return lambda orientation: np.array(
            (np.cos(orientation), np.sin(orientation)))
    elif isinstance(orientation, Array):
        return lambda orientation: np.vstack(
            (np.cos(orientation), np.sin(orientation))).T
    else:
        raise TypeError('')


@numba.jit([f8[:](f8[:], f8[:], f8),
            f8[:, :](f8[:, :], f8[:, :], f8)],
           nopython=True, nogil=True, cache=True)
def weighted_average(e0, e1, weight):
    r"""Weighted average of two (unit)vectors

    .. math::
       \mathbf{\hat{e}}_{out} =
       \mathcal{N} \big(p \mathbf{\hat{e}_{0}} + (1 - p) \mathbf{\hat{e}_{1}} \big)

    where

    - :math:`\mathcal{N}(\mathbf{v}) = \frac{\mathbf{v}}{\|\mathbf{v}\|}` is the
      normalization of the vector

    Args:
        e0 (numpy.ndarray): Unit vector :math:`\mathbf{\hat{e}_{0}}`
        e1 (numpy.ndarray): Unit vector :math:`\mathbf{\hat{e}_{1}}`
        weight (float): Weight between :math:`p \in [0, 1]`

    Returns:
        numpy.ndarray:
    """
    return weight * e0 + (1 - weight) * e1
