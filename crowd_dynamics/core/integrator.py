import numpy as np

from .force import force_adjust, force_random
from .interactions import agent_agent, agent_wall
from .torque import torque_adjust, torque_random


def motion(constant, agent, walls):
    # Target direction and angle
    agent.goal_to_target_direction()
    if agent.orientable_flag:
        agent.velocity_to_target_angle()

    # Motion
    agent.reset()

    force_adjust(constant, agent)
    force_random(constant, agent)

    if agent.orientable_flag:
        torque_adjust(constant, agent)
        torque_random(agent)

    agent_agent(constant, agent)

    for wall in walls:
        agent_wall(constant, agent, wall)


def integrator(result, constant, agent, walls):
    """Explicit euler method"""
    motion(constant, agent, walls)

    # TODO: Adaptive time step from maximum position change for agent.
    # Raise warning if using less than minimum step size
    # Larger crowd densities requires smaller timestep
    acceleration = agent.force / agent.mass

    # Position change
    dv = agent.velocity + acceleration * constant.dt
    dt = constant.dx_max / np.max(np.hypot(dv[:, 0], dv[:, 1]))
    if dt > constant.dt:
        dt = constant.dt
    elif dt < constant.dt_min:
        dt = constant.dt_min

    agent.velocity += acceleration * dt
    agent.position += agent.velocity * dt

    if agent.orientable_flag:
        angular_acceleration = agent.torque / agent.inertia_rot
        agent.angular_velocity += angular_acceleration * dt
        agent.angle += agent.angular_velocity * dt

    # Save
    result.increment_simulation_time(dt)
