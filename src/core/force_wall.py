import numba

from src.core.force import force_contact, force_social
from src.core.functions import rotate270, force_limit


@numba.jit(nopython=True, nogil=True)
def f_agent_wall(constant, agent, wall):
    for w in range(wall.size):
        for i in range(agent.size):
            distance, normal = wall.distance_with_normal(w, agent.position[i])
            radius = agent.radius[i, 0]
            relative_distance = radius - distance

            if distance <= agent.sight_wall:
                relative_position = wall.relative_position(w, agent.position[i],
                                                           agent.velocity[i])
                force = force_social(relative_position,
                                     agent.velocity[i],
                                     radius,
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
