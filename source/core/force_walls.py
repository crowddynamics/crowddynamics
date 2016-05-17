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


@numba.jit(nopython=True, nogil=True)
def f_iw(j, x_i, v_i, r_i, wall, constant):
    rot270 = np.array(((0.0, 1.0), (-1.0, 0.0)))
    force = np.zeros(2)

    d_iw, n_iw = wall.distance_with_normal(j, x_i)
    h_iw = r_i - d_iw

    if d_iw <= constant.sight:
        force += f_soc_iw(h_iw, n_iw, constant.a, constant.b)

    if h_iw > 0:
        t_iw = np.dot(rot270, n_iw)
        force += f_c_iw(v_i, t_iw, n_iw, h_iw, constant.mu, constant.kappa)

    return force


@numba.jit(nopython=True, nogil=True)
def f_iw_tot(constant, agent, wall):
    force = np.zeros(agent.shape)
    x = agent.position
    v = agent.velocity
    # TODO: Fix scalar vs array
    r = agent.radius.flatten()

    for i in range(agent.size):
        for j in range(wall.size):
            force[i] += f_iw(j, x[i], v[i], r[i], wall, constant)

    return force
