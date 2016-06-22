import numba
import numpy as np

from crowd_dynamics.core.force import force_social, force_contact, \
    force_social_velocity_independent
from crowd_dynamics.core.torque import torque
from crowd_dynamics.core.vector2d import length, truncate, rotate270


@numba.jit(nopython=True, nogil=True)
def agent_agent(agent):
    # n - 1 + n - 2 + ... + 1 = n^2 / 2
    for i in range(agent.size - 1):
        for j in range(i + 1, agent.size):
            agent_agent_interaction(i, j, agent)


@numba.jit(nopython=True, nogil=True)
def agent_wall(agent, wall):
    for w in range(wall.size):
        for i in range(agent.size):
            agent_wall_interaction(i, w, agent, wall)


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
            d = length(x)
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
def agent_agent_interaction(i, j, agent):
    # Function params
    x = agent.position[i] - agent.position[j]  # Relative positions
    r_tot = agent.radius[i] + agent.radius[j]  # Total radius
    d = length(x)                              # Distance
    h = d - r_tot                              # Relative distance

    # Agent sees the other agent
    if h <= agent.sight_soc:
        v = agent.velocity[i] - agent.velocity[j]      # Relative velocity
        r_moment_i, r_moment_j = np.zeros(2), np.zeros(2)

        force = force_social(x, v, r_tot, agent.k, agent.tau_0)
        truncate(force, agent.f_soc_ij_max)

        # TODO: Cutoff distance.
        cutoff = 2.0
        if h <= cutoff:
            if agent.orientable_flag:
                # Update values
                n, h, r_moment_i, r_moment_j = agent_agent_distance(agent, i, j)
            else:
                n = x / d  # Normal vector

            # Physical contact
            if h < 0:
                t = rotate270(n)  # Tangent vector
                force_c = force_contact(h, n, v, t, agent.mu, agent.kappa)
                force += force_c

        agent.force[i] += force
        agent.force[j] -= force
        if agent.orientable_flag:
            agent.torque[i] += torque(r_moment_i, force)
            agent.torque[j] -= torque(r_moment_j, force)
        agent.force_agent[i] += force
        agent.force_agent[j] -= force


@numba.jit(nopython=True, nogil=True)
def agent_wall_interaction(i, w, agent, wall):
    # Function params
    x = agent.position[i]
    r_tot = agent.radius[i]
    d, n = wall.distance_with_normal(w, x)
    h = d - r_tot  # Relative distance

    if h <= agent.sight_wall:
        r_moment_i = np.zeros(2)
        force = force_social_velocity_independent(h, n, agent.a, agent.b)

        # TODO: Velocity relative social force for agent-wall interaction
        # x, r = wall.relative_position(w, agent.position[i], agent.velocity[i])
        # force = force_social(x, agent.velocity[i], agent.radius[i] + r,
        #                      constant.k, constant.tau_0)

        truncate(force, agent.f_soc_iw_max)

        if h <= 2.0:
            if agent.orientable_flag:
                h, n, r_moment_i = agent_wall_distance(agent, wall, i, w)

            if h < 0:
                t = rotate270(n)  # Tangent
                force_c = force_contact(h, n, agent.velocity[i], t, agent.mu,
                                        agent.kappa)
                force += force_c

        agent.force[i] += force
        agent.force_wall[i] += force
        if agent.orientable_flag:
            agent.torque[i] += torque(r_moment_i, force)