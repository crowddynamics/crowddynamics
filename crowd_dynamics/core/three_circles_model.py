import numba
import numpy as np

from .force import force_social, force_contact
from .vector2d import force_limit, rotate270, normalize


@numba.jit(nopython=True, nogil=True)
def agent_agent_distance(agent, i, j, x_tilde, d_ij):
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
            x = c_i - c_j + x_tilde
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
    relative_position = agent.position[i] - agent.position[j]
    relative_velocity = agent.velocity[i] - agent.velocity[j]
    total_radius = agent.radius[i] + agent.radius[j]
    distance = np.hypot(relative_position[0], relative_position[1])
    relative_distance = distance - total_radius

    # Agent sees the other agent
    if relative_distance <= agent.sight_soc:
        force_soc = force_social(relative_position,
                                 relative_velocity,
                                 total_radius,
                                 constant.k,
                                 constant.tau_0)
        force_limit(force_soc, constant.f_soc_ij_max)
        agent.force[i] += force_soc
        agent.force[j] -= force_soc
        agent.force_agent[i] += force_soc
        agent.force_agent[j] -= force_soc

    # Physical contact
    if relative_distance < 0:
        normal = relative_position / distance
        tangent = rotate270(normal)
        force_c = force_contact(relative_distance,
                                normal,
                                relative_velocity,
                                tangent,
                                constant.mu,
                                constant.kappa)
        force_limit(force_c, constant.f_c_ij_max)
        agent.force[i] += force_c
        agent.force[j] -= force_c
        agent.force_agent[i] += force_c
        agent.force_agent[j] -= force_c

    # Herding
    if agent.herding_flag and distance <= agent.sight_herding:
        agent.neighbor_direction[i] += normalize(agent.velocity[j])
        agent.neighbor_direction[j] += normalize(agent.velocity[i])
        agent.neighbors[i] += 1
        agent.neighbors[j] += 1


@numba.jit(nopython=True, nogil=True)
def agent_wall_interaction(i, w, constant, agent, wall):
    # Function params
    distance, normal = wall.distance_with_normal(w, agent.position[i])
    relative_distance = distance - agent.radius[i]

    if relative_distance <= agent.sight_wall:
        relative_position = wall.relative_position(
            w, agent.position[i], agent.velocity[i])
        force = force_social(relative_position,
                             agent.velocity[i],
                             agent.radius[i],
                             constant.k,
                             constant.tau_0)

        force_limit(force, constant.f_soc_iw_max)
        agent.force[i] += force
        agent.force_wall[i] += force

    if relative_distance < 0:
        t_iw = rotate270(normal)
        force = force_contact(relative_distance,
                              normal,
                              agent.velocity[i],
                              t_iw,
                              constant.mu,
                              constant.kappa)
        force_limit(force, constant.f_c_iw_max)
        agent.force[i] += force
        agent.force_wall[i] += force
