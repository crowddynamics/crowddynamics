import numba
import numpy as np
from numba import f8
from scipy.stats import truncnorm

from .vector2d import dot2d, wrap_to_pi


def force_random(agent, std_trunc=3.0):
    # Truncated normal distribution with standard deviation of 3.
    magnitude = truncnorm.rvs(0, std_trunc, loc=0, scale=agent.std_rand_force,
                              size=agent.size)
    angle = np.random.uniform(0, 2 * np.pi, size=agent.size)
    force = magnitude * np.array((np.cos(angle), np.sin(angle)))
    agent.force += force.T * agent.mass


def torque_random(agent, std_trunc=3.0):
    """Random torque."""
    torq = truncnorm.rvs(-std_trunc, std_trunc, loc=0,
                         scale=agent.std_rand_force, size=agent.size)
    agent.torque += torq * agent.inertia_rot


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


@numba.jit(f8[:](f8, f8[:], f8[:], f8[:], f8, f8, f8), nopython=True,
           nogil=True)
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
    tau_max = 30.0  # Maximum time for interaction.

    if tau <= 0 or tau > tau_max:
        return force

    # Force is returned negative as repulsive force
    force += - k / (a * tau ** 2.0) * np.exp(-tau / tau_0) * \
             (2.0 / tau + 1.0 / tau_0) * \
             (v_rel - (v_rel * b + x_rel * a) / d)

    return force


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


def motion(agent, walls):
    """
    Updates rotational and spatial equations of motion.

    :param agent: Agent class
    :param walls: Walls calss
    :return: None. Updates values of agent and walls.
    """
    from .interactions import agent_agent, agent_wall

    # TODO: close contact / high crowd density counterflow model
    agent.reset()  # Reset forces and torque
    force_adjust(agent)
    force_random(agent)
    if agent.orientable:
        torque_adjust(agent)
        torque_random(agent)
    agent_agent(agent)
    for wall in walls:
        agent_wall(agent, wall)
