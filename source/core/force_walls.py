import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def f_soc_iw(h_iw, n_iw, a_i, b_i):
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
    magnitude = a_i * np.exp(h_iw / b_i)
    f_max = 1 * a_i
    if magnitude > f_max:
        magnitude = f_max
    force = magnitude * n_iw
    return force


@numba.jit(nopython=True, nogil=True)
def f_c_iw(v_i, t_iw, n_iw, h_iw, mu, kappa):
    force = h_iw * (mu * n_iw - kappa * np.dot(v_i, t_iw) * t_iw)
    return force


# @numba.jit(nopython=True, nogil=True)
def f_iw_round():
    pass


@numba.jit(nopython=True, nogil=True)
def f_iw_linear(x_i, v_i, r_i, p_0, p_1, t_w, n_w, l_w, constant):
    rot270 = np.array(((0.0, 1.0), (-1.0, 0.0)))
    force = np.zeros(2)

    q_0 = x_i - p_0
    q_1 = x_i - p_1

    l_t = - np.dot(t_w, q_1) - np.dot(t_w, q_0)

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

    h_iw = r_i - d_iw

    if d_iw <= constant.sight:
        force += f_soc_iw(h_iw, n_iw, constant.a, constant.b)

    if h_iw > 0:
        t_iw = np.dot(rot270, n_iw)
        force += f_c_iw(v_i, t_iw, n_iw, h_iw, constant.mu, constant.kappa)

    return force


@numba.jit(nopython=True, nogil=True)
def f_iw_linear_tot(constant, agent, linear_wall):
    force = np.zeros((agent.size, 2))
    x = agent.position
    v = agent.velocity
    # TODO: Fix scalar vs array
    r = agent.radius.flatten()

    for i in range(agent.size):
        for j in range(linear_wall.size):
            p_0, p_1, t_w, n_w, l_w = linear_wall.deconstruct(j)
            force += f_iw_linear(x[i], v[i], r[i], p_0, p_1, t_w, n_w, l_w,
                                 constant)

    return force
