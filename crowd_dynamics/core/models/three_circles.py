import numba
import numpy as np

from crowd_dynamics.core.force import force_social, force_contact
from crowd_dynamics.core.torque import torque
from crowd_dynamics.core.vector2d import force_limit, rotate270, normalize


@numba.jit(nopython=True, nogil=True)
def agent_agent_distance(agent, i, j, d_ij):
    """Three circles model"""
    # Positions
    x_i = (agent.position[i], agent.position_ls[i], agent.position_rs[i])
    x_j = (agent.position[j], agent.position_ls[j], agent.position_rs[j])

    # Radii of torso and shoulders
    r_i = (agent.r_t[i], agent.r_s[i], agent.r_s[i])
    r_j = (agent.r_t[j], agent.r_s[j], agent.r_s[j])

    # Minimizing values
    positions = np.zeros(2), np.zeros(2)  #
    radius = (0.0, 0.0)                   # Radius
    relative_position = np.zeros(2)       # Vector from agent i s/t to agent j s/t
    relative_distance = d_ij              # Minimum relative distance distance
    direction = np.zeros(2)               # Unit vector of x_rel

    for xi, ri  in zip(x_i, r_i):
        for xj, rj in zip(x_j, r_j):
            x = xi - xj
            d = np.hypot(x[0], x[1])
            h = d - (ri + rj)
            if h < relative_distance:
                relative_distance = h
                radius = ri, rj
                relative_position = x
                direction = x / d
                positions = xi, xj

    r_moment_i = positions[0] + radius[0] * direction
    r_moment_j = positions[1] - radius[1] * direction

    return relative_position, relative_distance, r_moment_i, r_moment_j


def agent_wall_distance(agent, wall, i, w):
    pass


@numba.jit(nopython=True, nogil=True)
def agent_agent_interaction(i, j, constant, agent):
    # Function params
    x = agent.position[i] - agent.position[j]      # Relative positions
    r_tot = agent.radius[i] + agent.radius[j]  # Total radius
    d = np.hypot(x[0], x[1])                       # Distance
    h = d - r_tot                          # Relative distance

    # Agent sees the other agent
    if h <= agent.sight_soc:
        v = agent.velocity[i] - agent.velocity[j]      # Relative velocity
        r_moment_i, r_moment_j = np.zeros(2), np.zeros(2)

        force = force_social(x, v, r_tot, constant.k, constant.tau_0)
        force_limit(force, constant.f_soc_ij_max)

        # TODO: Cutoff distance.
        if h <= 2.0:
            x, h, r_moment_i, r_moment_j = agent_agent_distance(agent, i, j, d)

        # Physical contact
        if h < 0:
            n = x / d  # Normal vector
            t = rotate270(n)  # Tangent vector
            force_c = force_contact(h, n, v, t, constant.mu, constant.kappa)
            force_limit(force_c, constant.f_c_ij_max)
            force += force_c

        agent.force[i] += force
        agent.force[j] -= force
        agent.torque[i] += torque(r_moment_i, force)
        agent.torque[j] += torque(r_moment_j, -force)
        agent.force_agent[i] += force
        agent.force_agent[j] -= force

    # Herding
    if agent.herding_flag and h <= agent.sight_herding:
        agent.neighbor_direction[i] += normalize(agent.velocity[j])
        agent.neighbor_direction[j] += normalize(agent.velocity[i])
        agent.neighbors[i] += 1
        agent.neighbors[j] += 1


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
