import numba
import numpy as np

from .vector2d import wrap_to_pi


@numba.jit(nopython=True)
def integrator(agent, dt_min, dt_max):
    """
    Explicit euler method using adative timestep for integrating differential
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
        # TODO: Raise warning
        dt = dt_min

    # TODO: linearize integration
    agent.velocity += acceleration * dt
    agent.position += agent.velocity * dt

    if agent.orientable:
        angular_acceleration = agent.torque / agent.inertia_rot
        agent.angular_velocity += angular_acceleration * dt
        agent.angle += agent.angular_velocity * dt
        agent.angle[:] = wrap_to_pi(agent.angle)

        # TODO: Move somewhere else?
        agent.update_shoulder_positions()

    return dt

