import numba
import numpy as np
from numba import f8
from scipy.stats import truncnorm as tn

from crowd_dynamics.core.vector2d import length_nx2
from .vector2d import dot2d, wrap_to_pi


def force_random(agent):
    # Truncated normal distribution with standard deviation of 3.
    i = agent.indices()
    magnitude = tn.rvs(0, 3, loc=0, scale=agent.std_rand_force, size=i.size)
    angle = np.random.uniform(0, 2 * np.pi, size=i.size)
    force = magnitude * np.array((np.cos(angle), np.sin(angle)))
    agent.force[i] += force.T * agent.mass[i]


def torque_random(agent):
    """Random torque."""
    i = agent.indices()
    torque = tn.rvs(-3, 3, loc=0, scale=agent.std_rand_force, size=i.size)
    agent.torque[i] += torque * agent.inertia_rot[i]


@numba.jit(nopython=True, nogil=True)
def force_adjust(agent):
    """Force that adjust movement towards target direction."""
    for i in agent.indices():
        force = (agent.mass[i] / agent.tau_adj) * \
                (agent.target_velocity[i] * agent.target_direction[i] -
                 agent.velocity[i])
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
    """Verlet integration using adaptive timestep for integrating differential
    system.

    :param dt_min: Minimum timestep for adaptive integration
    :param dt_max: Maximum timestep for adaptive integration
    :param agent: Agent class
    :return: Timestep that was used for integration
    """
    i = agent.indices()
    acceleration = agent.force[i] / agent.mass[i]
    dv = agent.velocity[i] + acceleration * dt_max
    dx_max = 1.1 * np.max(agent.target_velocity[i]) * dt_max

    dv_max = np.max(length_nx2(dv))
    if dv_max == 0:
        dt = dt_max
    else:
        dt = dx_max / dv_max
        if dt > dt_max:
            dt = dt_max
        elif dt < dt_min:
            dt = dt_min

    agent.position[i] += agent.velocity[i] * dt + 0.5 * acceleration * dt ** 2
    agent.velocity[i] += acceleration * dt

    if agent.orientable:
        angular_acceleration = agent.torque[i] / agent.inertia_rot[i]
        agent.angle[i] += agent.angular_velocity[i] * dt + \
                          angular_acceleration * 0.5 * dt ** 2
        agent.angular_velocity[i] += angular_acceleration * dt
        agent.angle[:] = wrap_to_pi(agent.angle)

    agent.update_shoulder_positions()

    return dt
