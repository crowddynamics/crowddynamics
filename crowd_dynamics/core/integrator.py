import numba
import numpy as np


from .interactions import agent_agent, agent_wall
from .motion import force_adjust, torque_adjust
from .navigation import direction_to_target_angle
from .vector2d import wrap_to_pi


def motion(agent, walls):
    # TODO: Active/Inactive agents
    # TODO: Navigation

    # Target angle update policy
    if agent.orientable_flag:
        direction_to_target_angle(agent)

    # Reset forces and torque
    agent.reset()

    # Motion
    force_adjust(agent)

    if agent.orientable_flag:
        torque_adjust(agent)

    agent_agent(agent)

    for wall in walls:
        agent_wall(agent, wall)


@numba.jit(nopython=True)
def integrator(dt_min, dt_max, agent):
    """Explicit euler method"""
    # TODO: Adaptive time step from maximum position change for agent.
    # Raise warning if using less than minimum step size
    # Larger crowd densities requires smaller timestep
    acceleration = agent.force / agent.mass

    # Position change
    # TODO:
    dv = agent.velocity + acceleration * dt_max
    dx_max = 5.0 * dt_max  # Max of agent.target_velocity = 5.0
    dt = dx_max / np.max(np.hypot(dv[:, 0], dv[:, 1]))
    # dt = dx_max / length_nx2(dv)
    if dt > dt_max:
        dt = dt_max
    elif dt < dt_min:
        dt = dt_min

    agent.velocity += acceleration * dt
    agent.position += agent.velocity * dt

    if agent.orientable_flag:
        angular_acceleration = agent.torque / agent.inertia_rot
        agent.angular_velocity += angular_acceleration * dt
        agent.angle += agent.angular_velocity * dt
        agent.angle = wrap_to_pi(agent.angle)

    if agent.orientable_flag:
        agent.update_shoulder_positions()

    return dt

