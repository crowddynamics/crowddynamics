import numba
import numpy as np

from crowd_dynamics.core.force import force_social, force_contact
from crowd_dynamics.core.functions import force_limit, rotate270, normalize


@numba.jit(nopython=True, nogil=True)
def agent_agent(constant, agent):
    # n - 1 + n - 2 + ... + 1 = n^2 / 2
    for i in range(agent.size - 1):
        for j in range(i + 1, agent.size):
            # Function params
            relative_position = agent.position[i] - agent.position[j]
            relative_velocity = agent.velocity[i] - agent.velocity[j]
            total_radius = agent.radius[i] + agent.radius[j]
            distance = np.hypot(relative_position[0], relative_position[1])
            relative_distance = distance - total_radius

            # If agent is orientable
            if agent.orientable_flag:
                # Update relative position, relative distance, total radius
                # and radius vectors for torque
                # Three circles model
                pass

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
def agent_wall(constant, agent, wall):
    for w in range(wall.size):
        for i in range(agent.size):
            # Function params
            distance, normal = wall.distance_with_normal(w, agent.position[i])
            relative_distance = distance - agent.radius[i]

            if relative_distance <= agent.sight_wall:
                relative_position = wall.relative_position(w, agent.position[i],
                                                           agent.velocity[i])
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
