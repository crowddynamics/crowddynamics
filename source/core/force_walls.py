import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def f_soc_iw(r_i, d_iw, n_iw, a_i, b_i):
    """
    Params
    ------
    :param a_i: Coefficient
    :param b_i: Coefficient
    :param r_i: Radius of the agent
    :param d_iw: Distance to the wall
    :param n_iw: Unit vector that is perpendicular to the agent and the wall
    :return:
    """
    force = a_i * np.exp((r_i - d_iw) / b_i) * n_iw
    return force


@numba.jit(nopython=True, nogil=True)
def f_c_iw(h_iw, n_iw, v_i, t_iw, mu, kappa):
    """

    :param h_iw:
    :param n_iw:
    :param v_i:
    :param t_iw:
    :param mu:
    :param kappa:
    :return:
    """
    force = h_iw * (mu * n_iw - kappa * np.dot(v_i, t_iw) * t_iw)
    return force


# @numba.jit(nopython=True, nogil=True)
def f_iw_linear(x_i, v_i, r_i, p_0, p_1, t_w, n_w, l_w, sight, a, b, mu, kappa):
    """

    :param x_i:
    :param v_i:
    :param r_i:
    :param p_0:
    :param p_1:
    :param t_w:
    :param n_w:
    :param l_w:
    :param sight:
    :param a:
    :param b:
    :param mu:
    :param kappa:
    :return:
    """
    force = np.zeros(2)

    q_0 = x_i - p_0
    q_1 = x_i - p_1

    l_t = np.dot(t_w, q_1) + np.dot(t_w, q_0)
    l_t *= -1

    if l_t > l_w:
        d_iw = np.hypot(q_0[0], q_0[1])
        n_iw = q_0 / d_iw
    elif l_t < -l_w:
        d_iw = np.hypot(q_1[0], q_1[1])
        n_iw = q_1 / d_iw
    else:
        l_n = np.dot(n_w, q_0)
        d_iw = np.abs(l_n)
        n_iw = np.sign(l_n) * n_w

    if d_iw <= sight:
        force += f_soc_iw(r_i, d_iw, n_iw, a, b)

    h_iw = r_i - d_iw
    if h_iw > 0:
        rot270 = np.array(((0, 1), (-1, 0)), dtype=np.float64)
        t_iw = np.dot(rot270, n_iw)
        force += f_c_iw(h_iw, n_iw, v_i, t_iw, mu, kappa)

    return force


@numba.jit(nopython=True, nogil=True)
def deconstruct_linear_wall(w):
    p_0 = w[0:2]
    p_1 = w[2:4]
    t_w = w[4:6]
    n_w = w[6:8]
    l_w = w[8]
    return p_0, p_1, t_w, n_w, l_w


# @numba.jit(nopython=True, nogil=True)
def f_iw_linear_tot(i, x, v, r, linear_wall, f_max, sight, mu, kappa, a, b):
    force = np.zeros(2)

    for row in range(len(linear_wall)):
        w = linear_wall[row]
        p_0, p_1, t_w, n_w, l_w = deconstruct_linear_wall(w)
        force += f_iw_linear(x[i], v[i], r[i], p_0, p_1, t_w, n_w, l_w,
                             sight, a, b, mu, kappa)

    return force
