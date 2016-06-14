import numba
import numpy as np

from .force import force_social, force_contact
from .torque import torque
from .vector2d import force_limit, rotate270, normalize


@numba.jit(nopython=True, nogil=True)
def agent_agent_distance(agent, i, j, x_rel_torso, d_ij):
    """Three circles model"""
    # TODO: Pre-compute tangents
    t_i = np.array((-np.sin(agent.angle[i]), np.cos(agent.angle[i])))
    t_j = np.array((-np.sin(agent.angle[j]), np.cos(agent.angle[j])))

    tr_i = agent.radius_torso_shoulder[i] * t_i
    tr_j = agent.radius_torso_shoulder[j] * t_j

    r_i = (agent.radius_torso[i],
           agent.radius_shoulder[i],
           agent.radius_shoulder[i])
    r_j = (agent.radius_torso[j],
           agent.radius_shoulder[j],
           agent.radius_shoulder[j])

    k = (0.0, 1.0, -1.0)
    r = (0.0, 0.0)
    c = (np.zeros(2), np.zeros(2))
    x_rel = np.zeros(2)
    e_ij = np.zeros(2)
    h_min = d_ij
    for ri, k_i in zip(r_i, k):
        for rj, k_j in zip(r_j, k):
            c_i = k_i * tr_i
            c_j = k_j * tr_j
            x = c_i - c_j + x_rel_torso
            d = np.hypot(x[0], x[1])
            h = d - (ri + rj)
            if h < h_min:
                h_min = h
                r = ri, rj
                c = c_i, c_j
                x_rel = x
                e_ij = x / d

    r_tot = r[0] + r[1]
    r_moment_i = agent.position[i] + c[0] + r[0] * e_ij
    r_moment_j = agent.position[j] + c[1] - r[1] * e_ij

    return x_rel, r_tot, h_min, r_moment_i, r_moment_j


def agent_wall_distance():
    pass


@numba.jit(nopython=True, nogil=True)
def agent_agent_interaction(i, j, constant, agent):
    # Function params
    x = agent.position[i] - agent.position[j]      # Relative positions
    v = agent.velocity[i] - agent.velocity[j]      # Relative velocity
    r_tot_max = agent.radius[i] + agent.radius[j]  # Total radius
    d = np.hypot(x[0], x[1])                       # Distance
    h_min = d - r_tot_max                          # Relative distance

    x, r_tot, h, r_moment_i, r_moment_j = agent_agent_distance(agent, i, j, x, d)

    # Agent sees the other agent
    if h <= agent.sight_soc:
        force_soc = force_social(x, v, r_tot, constant.k, constant.tau_0)
        force_limit(force_soc, constant.f_soc_ij_max)
        agent.force[i] += force_soc
        agent.force[j] -= force_soc
        agent.torque[i] += torque(r_moment_i, force_soc)
        agent.torque[j] += torque(r_moment_j, force_soc)
        agent.force_agent[i] += force_soc
        agent.force_agent[j] -= force_soc

    # Physical contact
    if h < 0:
        n = x / d  # Normal vector
        t = rotate270(n)  # Tangent vector
        force_c = force_contact(h, n, v, t, constant.mu, constant.kappa)
        force_limit(force_c, constant.f_c_ij_max)
        agent.force[i] += force_c
        agent.force[j] -= force_c
        agent.torque[i] += torque(r_moment_i, force_c)
        agent.torque[j] += torque(r_moment_j, force_c)
        agent.force_agent[i] += force_c
        agent.force_agent[j] -= force_c

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
        agent.force[i] += force
        agent.force_wall[i] += force

    if h < 0:
        t = rotate270(n)  # Tangent
        force = force_contact(h, n, agent.velocity[i], t, constant.mu,
                              constant.kappa)
        force_limit(force, constant.f_c_iw_max)
        agent.force[i] += force
        agent.force_wall[i] += force
