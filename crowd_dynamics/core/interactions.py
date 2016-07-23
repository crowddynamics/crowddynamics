import numba
import numpy as np

from crowd_dynamics.core.vector2d import cross2d
from .motion import force_social, force_social_velocity_independent, \
    force_contact
from .vector2d import length, truncate, rotate270


@numba.jit(nopython=True, nogil=True)
def agent_agent(agent):
    # n - 1 + n - 2 + ... + 1 = n^2 / 2 in O(n^2)
    ind = agent.indices()
    for l, i in enumerate(ind[:-1]):
        for j in ind[l + 1:]:
            agent_agent_interaction(i, j, agent)


@numba.jit(nopython=True, nogil=True)
def agent_wall(agent, wall):
    ind = agent.indices()
    for w in range(wall.size):
        for i in ind:
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
    positions = np.zeros(agent.shape[1]), np.zeros(agent.shape[1])  #
    radius = (0.0, 0.0)  # Radius
    relative_distance = np.nan  # Minimum relative distance distance
    normal = np.zeros(agent.shape[1])  # Unit vector of x_rel

    for xi, ri in zip(x_i, r_i):
        for xj, rj in zip(x_j, r_j):
            x = xi - xj
            d = length(x)
            r_tot = (ri + rj)
            h = d - r_tot
            if np.isnan(relative_distance) or h < relative_distance:
                relative_distance = h
                radius = ri, rj
                normal = x / d
                positions = xi, xj

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
    d = length(x)  # Distance
    h = d - r_tot  # Relative distance

    # Agent sees the other agent
    if h <= agent.sight_soc:
        v = agent.velocity[i] - agent.velocity[j]  # Relative velocity
        r_moment_i, r_moment_j = np.zeros(2), np.zeros(2)
        # force_i, force_j = np.zeros(2), np.zeros(2)
        force = force_social(x, v, r_tot, agent.mean_mass, agent.k_soc,
                             agent.tau_0, agent.f_soc_ij_max)

        if h <= agent.dist_three_circle:
            if agent.orientable:
                # Update values
                n, h, r_moment_i, r_moment_j = agent_agent_distance(agent, i, j)
                # force
            else:
                n = x / d  # Normal vector
                # force = force_social(x, v, r_tot, agent.mean_mass, agent.k_soc,
                #                      agent.tau_0, agent.f_soc_ij_max)
                # force_i[:] = force
                # force_j[:] = force

            # Physical contact
            if h < 0:
                t = rotate270(n)  # Tangent vector
                force_c = force_contact(h, n, v, t, agent.mu, agent.kappa,
                                        agent.damping)
                force += force_c

        agent.force[i] += force
        agent.force[j] -= force
        if agent.orientable:
            agent.torque[i] += cross2d(r_moment_i, force)
            agent.torque[j] -= cross2d(r_moment_j, force)

    # TODO: update neighborhood
    if agent.neighbor_radius > 0 and h < agent.neighbor_radius:
        if h < agent.neighbor_distances_max[i]:
            ind = np.argmax(agent.neighbor_distances[i])
            agent.neighbors[i, ind] = j
            agent.neighbor_distances[i, ind] = h
            agent.neighbor_distances_max[i] = np.max(
                agent.neighbor_distances[i])

        if h < agent.neighbor_distances_max[j]:
            ind = np.argmax(agent.neighbor_distances[j])
            agent.neighbors[j, ind] = i
            agent.neighbor_distances[j, ind] = h
            agent.neighbor_distances_max[j] = np.max(
                agent.neighbor_distances[j])


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
        truncate(force, agent.f_soc_iw_max)

        # TODO: Velocity relative social force for agent-wall interaction
        # x, r = wall.relative_position(w, agent.position[i], agent.velocity[i])
        # force = force_social(x, agent.velocity[i], agent.radius[i] + r,
        #                      constant.k_soc, constant.tau_0)

        if h <= agent.dist_three_circle:
            if agent.orientable:
                h, n, r_moment_i = agent_wall_distance(agent, wall, i, w)

            if h < 0:
                t = rotate270(n)  # Tangent
                force_c = force_contact(h, n, agent.velocity[i], t, agent.mu,
                                        agent.kappa, agent.damping)
                force += force_c

        agent.force[i] += force
        if agent.orientable:
            agent.torque[i] += cross2d(r_moment_i, force)
