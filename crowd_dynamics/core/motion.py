import numba
import numpy as np
from numba import f8
from scipy.stats import truncnorm

from .vector2d import dot2d, wrap_to_pi


def force_random(agent, std_trunc=3.0):
    # Truncated normal distribution with standard deviation of 3.
    i = agent.indices()
    magnitude = truncnorm.rvs(0, std_trunc, loc=0, scale=agent.std_rand_force,
                              size=i.size)
    angle = np.random.uniform(0, 2 * np.pi, size=i.size)
    force = magnitude * np.array((np.cos(angle), np.sin(angle)))
    agent.force[i] += force.T * agent.mass[i]


def torque_random(agent, std_trunc=3.0):
    """Random torque."""
    i = agent.indices()
    torque = truncnorm.rvs(-std_trunc, std_trunc, loc=0,
                           scale=agent.std_rand_force, size=i.size)
    agent.torque[i] += torque * agent.inertia_rot[i]


@numba.jit(nopython=True, nogil=True)
def force_adjust(agent):
    """Force that adjust movement towards target direction."""
    for i in agent.indices():
        force = (agent.mass[i] / agent.tau_adj) * \
                (agent.target_velocity[i] * agent.target_direction[i] - agent.velocity[i])
        agent.force[i] += force
        # agent.force_adjust += force


@numba.jit(nopython=True, nogil=True)
def torque_adjust(agent):
    """Adjusting torque."""
    for i in agent.indices():
        agent.torque[i] += agent.inertia_rot[i] / agent.tau_adj_rot * (
            wrap_to_pi(agent.target_angle[i] - agent.angle[i]) / np.pi *
            agent.target_angular_velocity[i] - agent.angular_velocity[i])


@numba.jit(f8[:](f8, f8[:], f8, f8), nopython=True, nogil=True)
def force_social_velocity_independent(h, n, a, b):
    """Simple velocity independent social force."""
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


@numba.jit(f8[:](f8, f8[:], f8[:], f8[:], f8, f8, f8), nopython=True,
           nogil=True)
def force_contact(h, n, v, t, mu, kappa, damping):
    """Frictional contact force with damping."""
    return - h * (mu * n - kappa * dot2d(v, t) * t) + damping * dot2d(v, n) * n


def motion(agent, walls):
    """
    Updates rotational and spatial equations of motion.

    :param agent: Agent class
    :param walls: Walls calss
    :return: None. Updates values of agent and walls.
    """
    from .interactions import agent_agent, agent_wall

    # TODO: close contact / high crowd density counterflow model
    agent.reset_motion()  # Reset forces and torque
    force_adjust(agent)
    force_random(agent)
    if agent.orientable:
        torque_adjust(agent)
        torque_random(agent)
    agent_agent(agent)
    for wall in walls:
        agent_wall(agent, wall)


@numba.jit(nopython=True)
def integrator(agent, dt_min, dt_max):
    """
    Verlet integration using adative timestep for integrating differential
    system.

    :param dt_min: Minimum timestep for adaptive integration
    :param dt_max: Maximum timestep for adaptive integration
    :param agent: Agent class
    :return: Timestep that was used for integration
    """
    # Larger crowd densities may require smaller timestep
    acceleration = agent.force / agent.mass

    dv = agent.velocity + acceleration * dt_max
    dx_max = 1.1 * np.max(agent.target_velocity) * dt_max
    dt = dx_max / np.max(np.hypot(dv[:, 0], dv[:, 1]))

    if dt > dt_max:
        dt = dt_max
    elif dt < dt_min:
        # TODO: Raise warning?
        dt = dt_min

    agent.position += agent.velocity * dt + 0.5 * acceleration * dt**2
    agent.velocity += acceleration * dt

    if agent.orientable:
        angular_acceleration = agent.torque / agent.inertia_rot
        agent.angle += agent.angular_velocity * dt + 0.5 * angular_acceleration * dt**2
        agent.angular_velocity += angular_acceleration * dt
        agent.angle[:] = wrap_to_pi(agent.angle)

    # TODO: Move somewhere else?
    agent.update_shoulder_positions()

    return dt
