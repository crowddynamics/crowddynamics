import numba
from numba import f8
import numpy as np

from .vector2d import dot2d, wrap_to_pi


@numba.jit(nopython=True, nogil=True)
def force_random(agent):
    """Random force"""
    for i in range(agent.size):
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(0, agent.f_random_fluctuation_max)
        agent.force[i][0] += magnitude * np.cos(angle)
        agent.force[i][1] += magnitude * np.sin(angle)


@numba.jit(nopython=True, nogil=True)
def force_adjust(agent):
    """Force that adjust movement towards target direction."""
    force = (agent.mass / agent.tau_adj) * \
            (agent.target_velocity * agent.target_direction - agent.velocity)
    agent.force += force
    agent.force_adjust += force


@numba.jit(f8[:](f8, f8[:], f8[:], f8[:], f8, f8), nopython=True, nogil=True)
def force_contact(h, n, v, t, mu, kappa):
    """Frictional contact force."""
    return - h * (mu * n - kappa * dot2d(v, t) * t)


@numba.jit(f8[:](f8, f8[:], f8[:], f8[:], f8, f8, f8), nopython=True, nogil=True)
def force_contact_damped(h, n, v, t, mu, kappa, damping):
    """Frictional contact force with damping."""
    return - h * (mu * n - kappa * dot2d(v, t) * t) + damping * dot2d(v, n) * n


@numba.jit(f8[:](f8, f8[:], f8, f8), nopython=True, nogil=True)
def force_social_velocity_independent(h, n, a, b):
    """Naive velocity independent social force."""
    return np.exp(- h / b) * a * n


@numba.jit(f8[:](f8[:], f8[:], f8, f8, f8), nopython=True, nogil=True)
def force_social(x_rel, v_rel, r_tot, k, tau_0):
    """
    Velocity dependent social force based on human anticipatory behaviour.
    http://motion.cs.umn.edu/PowerLaw/
    """
    force = np.zeros_like(x_rel)

    a = dot2d(v_rel, v_rel)
    b = -dot2d(x_rel, v_rel)
    c = dot2d(x_rel, x_rel) - r_tot ** 2
    d = np.sqrt(b ** 2 - a * c)

    # Avoid zero division.
    # No interaction if tau cannot be defined.
    if np.isnan(d) or d == 0 or a == 0:
        return force

    tau = (b - d) / a  # Time-to-collision. In seconds
    tau_max = 30.0     # Maximum time for interaction.

    if tau <= 0 or tau > tau_max:
        return force

    # Force is returned negative as repulsive force
    force += - k / (a * tau ** 2.0) * np.exp(-tau / tau_0) * \
             (2.0 / tau + 1.0 / tau_0) * \
             (v_rel - (v_rel * b + x_rel * a) / d)

    return force


@numba.jit(nopython=True, nogil=True)
def torque_random(agent):
    """Random torque."""
    for i in range(agent.size):
        agent.torque[i] += np.random.uniform(-1, 1)


@numba.jit(nopython=True, nogil=True)
def torque_adjust(agent):
    """Adjusting torque."""
    agent.torque += agent.inertia_rot / agent.tau_adj_rot * (
        wrap_to_pi(agent.target_angle - agent.angle) / np.pi *
        agent.target_angular_velocity - agent.angular_velocity)


@numba.jit(nopython=True, nogil=True)
def torque(radius, force):
    """Torque for 2D vectors. Right corner from cross product."""
    return radius[0] * force[1] - radius[1] * force[0]
