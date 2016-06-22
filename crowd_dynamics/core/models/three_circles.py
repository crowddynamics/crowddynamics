import numba
import numpy as np

from crowd_dynamics.core.force import force_social, force_contact, \
    force_social_velocity_independent
from crowd_dynamics.core.torque import torque
from crowd_dynamics.core.vector2d import truncate, rotate270


@numba.jit(nopython=True, nogil=True)
def agent_agent_distance(agent, i, j):
    # Positions: center, left, right
    x_i = (agent.position[i], agent.position_ls[i], agent.position_rs[i])
    x_j = (agent.position[j], agent.position_ls[j], agent.position_rs[j])

    # Radii of torso and shoulders
    r_i = (agent.r_t[i], agent.r_s[i], agent.r_s[i])
    r_j = (agent.r_t[j], agent.r_s[j], agent.r_s[j])

    # Minimizing values
    positions = np.zeros(2), np.zeros(2)  #
    radius = (0.0, 0.0)                   # Radius
    relative_distance = 0                 # Minimum relative distance distance
    normal = np.zeros(2)                  # Unit vector of x_rel

    init = True
    for xi, ri in zip(x_i, r_i):
        for xj, rj in zip(x_j, r_j):
            x = xi - xj
            d = np.hypot(x[0], x[1])
            r_tot = (ri + rj)
            h = d - r_tot
            if h < relative_distance or init:
                relative_distance = h
                radius = ri, rj
                normal = x / d
                positions = xi, xj
                init = False

    r_moment_i = positions[0] + radius[0] * normal - agent.position[i]
    r_moment_j = positions[1] - radius[1] * normal - agent.position[j]

    return normal, relative_distance, r_moment_i, r_moment_j


@numba.jit(nopython=True, nogil=True)
def agent_wall_distance(agent, wall, i, w):
    x_i = (agent.position[i], agent.position_ls[i], agent.position_rs[i])
    r_i = (agent.r_t[i], agent.r_s[i], agent.r_s[i])

    relative_distance = 0
    position = np.zeros(2)
    normal = np.zeros(2)
    radius = 0
    init = True
    for xi, ri in zip(x_i, r_i):
        d, n = wall.distance_with_normal(w, xi)
        h = d - ri
        if h < relative_distance or init:
            position = xi
            radius = ri
            relative_distance = h
            normal = n
            init = False

    r_moment_i = position - radius * normal - agent.position[i]

    return relative_distance, normal, r_moment_i


@numba.jit(nopython=True, nogil=True)
def agent_agent_interaction(i, j, constant, agent):
    # Function params
    x = agent.position[i] - agent.position[j]  # Relative positions
    r_tot = agent.radius[i] + agent.radius[j]  # Total radius
    d = np.hypot(x[0], x[1])                   # Distance
    h = d - r_tot                              # Relative distance

    # Agent sees the other agent
    if h <= agent.sight_soc:
        v = agent.velocity[i] - agent.velocity[j]      # Relative velocity
        r_moment_i, r_moment_j = np.zeros(2), np.zeros(2)

        force = force_social(x, v, r_tot, constant.k, constant.tau_0)
        truncate(force, constant.f_soc_ij_max)

        # TODO: Cutoff distance.
        cutoff = 2.0
        if h <= cutoff:
            n, h, r_moment_i, r_moment_j = agent_agent_distance(agent, i, j)

            # Physical contact
            if h < 0:
                # n = x / d  # Normal vector
                t = rotate270(n)  # Tangent vector
                force_c = force_contact(h, n, v, t, constant.mu, constant.kappa)
                truncate(force_c, constant.f_c_ij_max)
                force += force_c

        agent.force[i] += force
        agent.force[j] -= force
        agent.torque[i] += torque(r_moment_i, force)
        agent.torque[j] -= torque(r_moment_j, force)
        agent.force_agent[i] += force
        agent.force_agent[j] -= force


@numba.jit(nopython=True, nogil=True)
def agent_wall_interaction(i, w, constant, agent, wall):
    # Function params
    x = agent.position[i]
    r_tot = agent.radius[i]
    d, n = wall.distance_with_normal(w, x)
    h = d - r_tot  # Relative distance

    if h <= agent.sight_wall:
        r_moment_i = np.zeros(2)
        # x, r = wall.relative_position(w, agent.position[i], agent.velocity[i])
        # force = force_social(x, agent.velocity[i], agent.radius[i] + r,
        #                      constant.k, constant.tau_0)
        force = force_social_velocity_independent(h, n, constant.a, constant.b)
        truncate(force, constant.f_soc_iw_max)

        if h <= 2.0:
            h, n, r_moment_i = agent_wall_distance(agent, wall, i, w)

            if h < 0:
                t = rotate270(n)  # Tangent
                force_c = force_contact(h, n, agent.velocity[i], t, constant.mu,
                                        constant.kappa)
                truncate(force_c, constant.f_c_iw_max)
                force += force_c

        agent.force[i] += force
        agent.force_wall[i] += force
        agent.torque[i] += torque(r_moment_i, force)
