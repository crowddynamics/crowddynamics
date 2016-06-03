import numba
from numpy import hypot

from src.core.force import f_social_ij, f_contact
from src.core.functions import rotate270, normalize, force_limit


@numba.jit(nopython=True, nogil=True)
def f_agent_agent(constant, agent):
    # n - 1 + n - 2 + ... + 1 = n^2 / 2
    for i in range(agent.size - 1):
        for j in range(i + 1, agent.size):
            relative_position = agent.position[i] - agent.position[j]
            relative_velocity = agent.velocity[i] - agent.velocity[j]
            total_radius = agent.get_radius(i) + agent.get_radius(j)
            distance = hypot(relative_position[0], relative_position[1])
            relative_distance = total_radius - distance

            # If another agent is in range of sight_soc.
            if distance <= agent.sight_soc:
                force = f_social_ij(relative_position,
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
                force = f_contact(relative_distance,
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
