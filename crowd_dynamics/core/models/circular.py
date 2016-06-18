import numba
import numpy as np

from crowd_dynamics.core.force import force_social, force_contact
from crowd_dynamics.core.vector2d import force_limit, rotate270, normalize


@numba.jit(nopython=True, nogil=True)
def agent_agent_interaction(i, j, constant, agent):
    # Function params
    x = agent.position[i] - agent.position[j]  # Relative positions
    r_tot = agent.radius[i] + agent.radius[j]  # Total radius
    d = np.hypot(x[0], x[1])  # Distance
    h = d - r_tot  # Relative distance

    # Agent sees the other agent
    if h <= agent.sight_soc:
        v = agent.velocity[i] - agent.velocity[j]  # Relative velocity

        force = force_social(x, v, r_tot, constant.k, constant.tau_0)
        force_limit(force, constant.f_soc_ij_max)

        # Physical contact
        if h < 0:
            n = x / d  # Normal vector
            t = rotate270(n)  # Tangent vector
            force_c = force_contact(h, n, v, t, constant.mu, constant.kappa)
            force_limit(force_c, constant.f_c_ij_max)
            force += force_c

        agent.force[i] += force
        agent.force[j] -= force
        agent.force_agent[i] += force
        agent.force_agent[j] -= force


@numba.jit(nopython=True, nogil=True)
def agent_wall_interaction(i, w, constant, agent, wall):
    # Function params
    d, n = wall.distance_with_normal(w, agent.position[i])
    h = d - agent.radius[i]  # Relative distance

    if h <= agent.sight_wall:
        x, r = wall.relative_position(w, agent.position[i], agent.velocity[i])
        force = force_social(x, agent.velocity[i], agent.radius[i] + r,
                             constant.k, constant.tau_0)
        force_limit(force, constant.f_soc_iw_max)

        if h < 0:
            t = rotate270(n)  # Tangent
            force_c = force_contact(h, n, agent.velocity[i], t, constant.mu,
                                    constant.kappa)
            force_limit(force_c, constant.f_c_iw_max)
            force += force_c

        agent.force[i] += force
        agent.force_wall[i] += force
