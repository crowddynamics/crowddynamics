import numba
from numpy import dot, exp

from source.core.functions import rotate270


@numba.jit(nopython=True, nogil=True)
def f_soc_iw(h_iw, n_iw, a, b):
    return min(1.0, exp(h_iw / b)) * a * n_iw


@numba.jit(nopython=True, nogil=True)
def f_c_iw(v_i, t_iw, n_iw, h_iw, mu, kappa):
    return h_iw * (mu * n_iw - kappa * dot(v_i, t_iw) * t_iw)


@numba.jit(nopython=True, nogil=True)
def f_agent_wall(constant, agent, wall):
    for w in range(wall.size):
        for i in range(agent.size):
            d_iw, n_iw = wall.distance_with_normal(w, agent.position[i])
            h_iw = agent.get_radius(i) - d_iw

            if d_iw <= agent.sight_wall:
                agent.force[i] += f_soc_iw(h_iw, n_iw, constant.a, constant.b)

            if h_iw > 0:
                t_iw = rotate270(n_iw)
                agent.force[i] += f_c_iw(agent.velocity[i], t_iw, n_iw, h_iw,
                                         constant.mu, constant.kappa)
