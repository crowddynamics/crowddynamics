r"""Distance functions for potentials.

Some of the functions also compute rotational moments for computing torque like

.. math::
   \mathbf{M} = \mathbf{r}_{\mathrm{moment}} \times (\mathbf{f}_{}^{soc} + \mathbf{f}_{}^{c})

"""
import numba
import numpy as np
from numba import float64
from numba.types import Tuple, UniTuple
from crowddynamics.core.vector2D import length, rotate90, dot


@numba.jit([Tuple((float64, float64[:]))(float64[:], float64,
                                         float64[:], float64)],
           nopython=True, nogil=True, cache=True)
def distance_circles(x0, r0, x1, r1):
    r"""
    Skin-to-Skin distance :math:`h`  with normal :math:`\mathbf{\hat{n}}`
    between two circles.

    .. math::

       h &= \|\mathbf{x}_0 - \mathbf{x}_1\| - (r_0 + r_1) \\
       \mathbf{\hat{n}} &= \frac{\mathbf{x}_0 - \mathbf{x}_1}{\|\mathbf{x}_0 - \mathbf{x}_1\|}

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


@numba.jit([Tuple((float64, float64[:], float64[:], float64[:]))(
    UniTuple(float64[:], 3), UniTuple(float64, 3),
    UniTuple(float64[:], 3), UniTuple(float64, 3)
)],
           nopython=True, nogil=True, cache=True)
def distance_three_circles(x0, r0, x1, r1):
    r"""
    Skin-to-Skin distance :math:`h` with normal :math:`\mathbf{\hat{n}}` and
    rotational moments :math:`\mathbf{r}_{\mathrm{moment}_i}` between two three-circle
    models with.

    .. math::

       X_i &= \{\mathbf{x}_{torso}, \mathbf{x}_{left shoulder}, \mathbf{x}_{right shoulder} \} \\
       R_i &= \{r_{torso}, r_{shoulder}, r_{shoulder} \}

    .. math::
    
       h = \min_{(\mathbf{x}_0, r_0) \in (X_0, R_0)} \min_{(\mathbf{x}_1, r_1) \in (X_1, R_1)} \left( \|\mathbf{x}_0 - \mathbf{x}_1\| - (r_0 + r_1) \right)

    With minimized values of :math:`\mathbf{x}_0`, :math:`\mathbf{x}_1`,
    :math:`r_0` and :math:`r_1` compute

    .. math::

       \mathbf{\hat{n}} &= \frac{\mathbf{x}_0 - \mathbf{x}_1}{\|\mathbf{x}_0 - \mathbf{x}_1\|} \\
       \mathbf{r}_{\mathrm{moment}_0} &= \mathbf{x}_0 + r_0 \mathbf{\hat{n}} - \mathbf{x}_{\mathrm{torso}} \\
       \mathbf{r}_{\mathrm{moment}_1} &= \mathbf{x}_1 - r_1 \mathbf{\hat{n}} - \mathbf{x}_{\mathrm{torso}}

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
            h, n = distance_circles(xi, ri, xj, rj)
            if h < h_min or np.isnan(h_min):
                h_min = h
                normal = n
                i_min = i
                j_min = j

    r_moment0 = x0[i_min] + r0[i_min] * normal - x0[0]
    r_moment1 = x0[j_min] - r1[j_min] * normal - x1[0]

    return h_min, normal, r_moment0, r_moment1


@numba.jit([Tuple((float64, float64[:]))(float64[:], float64, float64[:], float64[:])],
           nopython=True, nogil=True, cache=True)
def distance_circle_line(x, r, p0, p1):
    r"""
    Skin-to-Skin distance between circle and line

    Args:
        x (numpy.ndarray):
        r (float):
        p0 (numpy.ndarray):
        p1 (numpy.ndarray):

    Returns:
        (float, numpy.ndarray): (skin-to-skin distance, normal vector)
    """
    # TODO: More docs
    d = p1 - p0
    l_w = length(d)
    t_w = d / l_w
    n_w = rotate90(t_w)

    q0 = x - p0
    q1 = x - p1
    l_t = - dot(t_w, q1) - dot(t_w, q0)

    if l_t > l_w:
        d_iw = length(q0)
        n_iw = q0 / d_iw
    elif l_t < -l_w:
        d_iw = length(q1)
        n_iw = q1 / d_iw
    else:
        l_n = dot(n_w, q0)
        d_iw = np.abs(l_n)
        n_iw = np.sign(l_n) * n_w

    h_iw = d_iw - r

    return h_iw, n_iw


@numba.jit([Tuple((float64, float64[:], float64[:]))(
    UniTuple(float64[:], 3), UniTuple(float64, 3),
    float64[:], float64[:]
)],
           nopython=True, nogil=True, cache=True)
def distance_three_circle_line(x, r, p0, p1):
    r"""
    Skin-to-Skin distance between three circle model and line

    Args:
        x ((numpy.ndarray, numpy.ndarray, numpy.ndarray)):
        r ((float, float, float)):
        p (numpy.ndarray):

    Returns:
        (float, numpy.ndarray, numpy.ndarray)
    """
    # TODO: More docs
    h_min = np.nan
    normal = np.zeros(2)
    i_min = 0

    for i, (x_, r_) in enumerate(zip(x, r)):
        h, n = distance_circle_line(x_, r_, p0, p1)
        if h < h_min or np.isnan(h_min):
            h_min = h
            normal = n
            i_min = i

    r_moment = x[i_min] - r[i_min] * normal - x[0]

    return h_min, normal, r_moment
