import numba
import numpy as np

from .force import force_social, force_contact
from .vector2d import force_limit, rotate270, normalize


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
