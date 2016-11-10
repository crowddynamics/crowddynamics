import numba
import numpy as np
from scipy.stats import truncnorm as tn

from .vector2D import wrap_to_pi, length_nx2


def force_fluctuation(agent):
    """
    Truncated normal distribution with standard deviation of 3.

    Args:
        agent (Agent):
    """
    i = agent.indices()
    magnitude = tn.rvs(0, 3, loc=0, scale=agent.std_rand_force, size=i.size)
    angle = np.random.uniform(0, 2 * np.pi, size=i.size)
    force = magnitude * np.array((np.cos(angle), np.sin(angle)))
    agent.force[i] += force.T * agent.mass[i]


def torque_fluctuation(agent):
    """
    Random torque.

    Args:
        agent (Agent):
    """
    if agent.orientable:
        i = agent.indices()
        torque = tn.rvs(-3, 3, loc=0, scale=agent.std_rand_force, size=i.size)
        agent.torque[i] += torque * agent.inertia_rot[i]


@numba.jit(nopython=True, nogil=True)
def force_adjust(agent):
    """
    Force that adjust movement towards target direction.

    Args:
        agent (Agent):
    """
    for i in agent.indices():
        force = (agent.mass[i] / agent.tau_adj) * \
                (agent.target_velocity[i] * agent.target_direction[i] -
                 agent.velocity[i])
        agent.force[i] += force


@numba.jit(nopython=True, nogil=True)
def torque_adjust(agent):
    """
    Adjusting torque.

    Args:
        agent (Agent):
    """
    if agent.orientable:
        for i in agent.indices():
            agent.torque[i] += agent.inertia_rot[i] / agent.tau_rot * (
                wrap_to_pi(agent.target_angle[i] - agent.angle[i]) / np.pi *
                agent.target_angular_velocity[i] - agent.angular_velocity[i])


@numba.jit(nopython=True)
def integrate(agent, dt_min, dt_max):
    """
    Verlet integration using adaptive timestep for integrating differential
    system.

    Args:
        agent (Agent):
        dt_min (float): Minimum timestep for adaptive integration
        dt_max (float): Maximum timestep for adaptive integration

    Returns:
        float: Timestep that was used for integration
    """
    i = agent.indices()
    a = agent.force[i] / agent.mass[i]  # Acceleration

    # Time step selection
    v_max = np.max(length_nx2(agent.velocity))
    dx_max = np.max(agent.target_velocity) * dt_max
    dx_max *= 1.1

    if v_max == 0:
        # Static system
        dt = dt_max
    else:
        dt = dx_max / v_max
        if dt > dt_max:
            dt = dt_max
        elif dt < dt_min:
            dt = dt_min

    # Updating agents
    agent.position[i] += agent.velocity[i] * dt + 0.5 * a * dt ** 2
    agent.velocity[i] += a * dt

    if agent.orientable:
        angular_acceleration = agent.torque[i] / agent.inertia_rot[i]
        agent.angle[i] += agent.angular_velocity[i] * dt + \
                          angular_acceleration * 0.5 * dt ** 2
        agent.angular_velocity[i] += angular_acceleration * dt
        agent.angle[:] = wrap_to_pi(agent.angle)

        agent.update_shoulder_positions()

    return dt
