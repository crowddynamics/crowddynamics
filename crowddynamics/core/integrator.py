import numba
import numpy as np

from crowddynamics.core.vector2D.vector2D import length_nx2, wrap_to_pi


@numba.jit(nopython=True)
def integrate(agent, dt_min, dt_max):
    r"""
    Integration using adaptive timestep for integrating differential
    system.

    Args:
        agent (Agent):
        dt_min (float): Minimum timestep for adaptive integration
        dt_max (float): Maximum timestep for adaptive integration

    Returns:
        float: Timestep that was used for integration
    """
    # TODO: Velocity Verlet?
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
