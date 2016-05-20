import numba
from numpy import sqrt, hypot, dot, exp, zeros_like, isnan, abs

from source.core.functions import rotate270, normalize, force_limit


@numba.jit(nopython=True, nogil=True)
def f_soc_ij(x_ij, v_ij, r_ij, k, tau_0):
    """
    About
    -----
    Social interaction force between two agents `i` and `j`. [1]

    References
    ----------
    [1] http://motion.cs.umn.edu/PowerLaw/
    """
    force = zeros_like(x_ij)

    a = dot(v_ij, v_ij)
    b = - dot(x_ij, v_ij)
    c = dot(x_ij, x_ij) - r_ij ** 2
    d = sqrt(b ** 2 - a * c)

    # Avoid zero division zero divisions.
    # No interaction if tau cannot be defined.
    if isnan(d) or d < 1.49e-08 or abs(a) < 1.49e-08:
        return force

    tau = (b - d) / a  # Time-to-collision
    tau_max = 999.0

    if tau <= 0 or tau > tau_max:
        return force

    # Force is returned negative as repulsive force
    m = 2.0  # Exponent in power law
    force -= k / (a * tau ** m) * exp(-tau / tau_0) * \
             (m / tau + 1 / tau_0) * (v_ij - (v_ij * b + x_ij * a) / d)

    return force


@numba.jit(nopython=True, nogil=True)
def f_c_ij(h_ij, n_ij, v_ij, t_ij, mu, kappa):
    return h_ij * (mu * n_ij - kappa * dot(v_ij, t_ij) * t_ij)


@numba.jit(nopython=True, nogil=True)
def f_agent_agent(constant, agent):
    for i in range(agent.size - 1):
        for j in range(i + 1, agent.size):
            relative_position = agent.position[i] - agent.position[j]
            relative_velocity = agent.velocity[i] - agent.velocity[j]
            total_radius = agent.get_radius(i) + agent.get_radius(j)
            distance = hypot(relative_position[0], relative_position[1])
            relative_distance = total_radius - distance

            # If another agent is in range of sight_soc.
            if distance <= agent.sight_soc:
                force = f_soc_ij(relative_position,
                                 relative_velocity,
                                 total_radius,
                                 constant.k,
                                 constant.tau_0)
                force_limit(force, constant.f_soc_ij_max)
                agent.force[i] += force
                agent.force[j] -= force

            # If agents are overlapping.
            if relative_distance > 0:
                normal = relative_position / distance
                tangent = rotate270(normal)
                force = f_c_ij(relative_distance,
                               normal,
                               relative_velocity,
                               tangent,
                               constant.mu,
                               constant.kappa)
                force_limit(force, constant.f_c_ij_max)
                agent.force[i] += force
                agent.force[j] -= force

            if agent.herding_flag and distance <= agent.sight_herding:
                agent.neighbor_direction[i] += normalize(agent.velocity[j])
                agent.neighbor_direction[j] += normalize(agent.velocity[i])
                agent.neighbors[i] += 1
                agent.neighbors[j] += 1
