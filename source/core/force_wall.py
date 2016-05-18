import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def f_soc_iw(h_iw, n_iw, a, b):
    """

    Params
    ------
    :param a: Coefficient
    :param b: Coefficient
    :param r_i: Radius of the agent
    :param d_iw: Distance to the wall
    :param n_iw: Unit vector that is perpendicular to the agent and the wall
    :return:
    """
    return min(1.0, np.exp(h_iw / b)) * a * n_iw


@numba.jit(nopython=True, nogil=True)
def f_c_iw(v_i, t_iw, n_iw, h_iw, mu, kappa):
    return h_iw * (mu * n_iw - kappa * np.dot(v_i, t_iw) * t_iw)


@numba.jit(nopython=True, nogil=True)
def f_iw_tot(constant, agent, wall):
    rot270 = np.array(((0.0, 1.0), (-1.0, 0.0)))
    force = np.zeros(agent.shape)

    for j in range(wall.size):
        for i in range(agent.size):
            d_iw, n_iw = wall.distance_with_normal(j, agent.position[i])
            h_iw = agent.get_radius(i) - d_iw

            if d_iw <= constant.sight:
                force[i] += f_soc_iw(h_iw, n_iw, constant.a, constant.b)

            if h_iw > 0:
                t_iw = np.dot(rot270, n_iw)
                force[i] += f_c_iw(agent.velocity[i], t_iw, n_iw, h_iw,
                                   constant.mu, constant.kappa)

    return force
