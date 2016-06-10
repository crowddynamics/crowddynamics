import numba
import numpy as np

from src.core.force import force_social, force_contact
from src.core.functions import force_limit, rotate270, normalize


@numba.jit(nopython=True, nogil=True)
def agent_agent(constant, agent):
    # n - 1 + n - 2 + ... + 1 = n^2 / 2
    for i in range(agent.size - 1):
        for j in range(i + 1, agent.size):
            relative_position = agent.position[i] - agent.position[j]
            relative_velocity = agent.velocity[i] - agent.velocity[j]
            total_radius = agent.radius[i] + agent.radius[j]
            distance = np.hypot(relative_position[0], relative_position[1])
            relative_distance = total_radius - distance

            # If agent is orientable
            # TODO: threshold distance
            if agent.orientable:
                # Update distance and calculate torque
                pass

            # If another agent is in range of sight_soc.
            if distance <= agent.sight_soc:
                force = force_social(relative_position,
                                     relative_velocity,
                                     total_radius,
                                     constant.k,
                                     constant.tau_0)
                force_limit(force, constant.f_soc_ij_max)
                agent.force[i] += force
                agent.force[j] -= force
                # agent.force_agent[i] += force
                # agent.force_agent[j] -= force

            # If agents are overlapping.
            if relative_distance > 0:
                normal = relative_position / distance
                tangent = rotate270(normal)
                force = force_contact(relative_distance,
                                      normal,
                                      relative_velocity,
                                      tangent,
                                      constant.mu,
                                      constant.kappa)
                force_limit(force, constant.f_c_ij_max)
                agent.force[i] += force
                agent.force[j] -= force
                # agent.force_agent[i] += force
                # agent.force_agent[j] -= force

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
            distance, normal = wall.distance_with_normal(w, agent.position[i])
            relative_distance = agent.radius[i] - distance

            if distance <= agent.sight_wall:
                relative_position = wall.relative_position(w, agent.position[i],
                                                           agent.velocity[i])
                force = force_social(relative_position,
                                     agent.velocity[i],
                                     agent.radius[i],
                                     constant.k,
                                     constant.tau_0)
                force_limit(force, constant.f_soc_iw_max)

                agent.force[i] += force
                # agent.force_wall[i] += force

            if relative_distance > 0:
                t_iw = rotate270(normal)
                force = force_contact(relative_distance,
                                      normal,
                                      agent.velocity[i],
                                      t_iw,
                                      constant.mu,
                                      constant.kappa)
                force_limit(force, constant.f_c_iw_max)
                agent.force[i] += force
                # agent.force_wall[i] += force