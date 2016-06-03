import numba
from numpy import dot, zeros_like, random, cos, pi, sin
from numpy.core.umath import exp, sqrt, isnan


@numba.jit(nopython=True, nogil=True)
def force_random_fluctuation(constant, agent):
    for i in range(agent.size):
        angle = random.uniform(0, 2 * pi)
        magnitude = random.uniform(0, constant.f_random_fluctuation_max)
        agent.force[i][0] += magnitude * cos(angle)
        agent.force[i][1] += magnitude * sin(angle)


@numba.jit(nopython=True, nogil=True)
def force_adjust(constant, agent):
    """
    Force that adjust movement towards target direction.
    """
    force = (agent.mass / constant.tau_adj) * \
            (agent.goal_velocity * agent.target_direction - agent.velocity)
    agent.force += force
    # agent.force_adjust += force


@numba.jit(nopython=True, nogil=True)
def force_social_naive(h_iw, n_iw, a, b):
    """
    Naive velocity independent social force.
    """
    return exp(h_iw / b) * a * n_iw


@numba.jit(nopython=True, nogil=True)
def force_contact(h, n, v, t, mu, kappa):
    """
    Frictional contact force.
    """
    return h * (mu * n - kappa * dot(v, t) * t)


@numba.jit(nopython=True, nogil=True)
def force_social(x_ij, v_ij, r_ij, k, tau_0):
    """
    Velocity dependent social force. [1]

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
    force -= k / (a * tau ** m) * exp(-tau / tau_0) * (m / tau + 1 / tau_0) * \
             (v_ij - (v_ij * b + x_ij * a) / d)

    return force
