import numba

from .force import force_adjust, force_random
from .interactions import agent_agent, agent_wall


# @numba.jit(nopython=True, nogil=True)
def explicit_euler_method(result, constant, agent, wall1, wall2):
    while True:
        # Target direction
        agent.goal_to_target_direction()
        # Target angle
        # agent.velocity_to_target_angle()

        # Motion
        agent.reset()
        force_adjust(constant, agent)
        # force_random(constant, agent)

        if agent.orientable_flag:
            # torque_adjust(constant, agent)
            # torque_random(agent)
            pass

        # Interactions
        agent_agent(constant, agent)
        agent_wall(constant, agent, wall1)
        agent_wall(constant, agent, wall2)

        # Integration
        acceleration = agent.force / agent.mass
        agent.velocity += acceleration * constant.dt
        agent.position += agent.velocity * constant.dt

        if agent.orientable_flag:
            angular_acceleration = agent.torque / agent.inertia_rot
            agent.angular_velocity += angular_acceleration * constant.dt
            agent.angle += agent.angular_velocity * constant.dt

        # Save
        result.increment_simu_time(constant.dt)
        yield


@numba.jit(nopython=True, nogil=True)
def _f_tot0(constant, agent):
    agent_agent(constant, agent)
    force_adjust(constant, agent)
    force_random(constant, agent)


@numba.jit(nopython=True, nogil=True)
def euler_method0(result, constant, agent):
    while True:
        # Target direction
        agent.goal_to_target_direction()
        # Update  position
        agent.reset()
        _f_tot0(constant, agent)
        acceleration = agent.force / agent.mass
        agent.velocity += acceleration * constant.dt
        agent.position += agent.velocity * constant.dt
        # Save
        result.increment_simu_time(constant.dt)
        yield


@numba.jit(nopython=True, nogil=True)
def _f_tot(constant, agent, wall):
    agent_agent(constant, agent)        # 53.4 %
    agent_wall(constant, agent, wall)   # 33.7 %
    force_adjust(constant, agent)              # 4.3 %
    force_random(constant, agent)  # 2.2 %


@numba.jit(nopython=True, nogil=True)
def euler_method(result, constant, agent, wall):
    while True:
        # Target direction
        agent.goal_to_target_direction()
        # Update  position
        agent.reset()
        _f_tot(constant, agent, wall)
        acceleration = agent.force / agent.mass
        agent.velocity += acceleration * constant.dt
        agent.position += agent.velocity * constant.dt
        # Save
        result.increment_simu_time(constant.dt)
        yield


@numba.jit(nopython=True, nogil=True)
def _f_tot2(constant, agent, wall1, wall2):
    agent_agent(constant, agent)
    agent_wall(constant, agent, wall1)
    agent_wall(constant, agent, wall2)
    force_adjust(constant, agent)
    force_random(constant, agent)


@numba.jit(nopython=True, nogil=True)
def euler_method2(result, constant, agent, wall1, wall2):
    while True:
        # Target direction
        agent.goal_to_target_direction()
        # Update  position
        agent.reset()
        _f_tot2(constant, agent, wall1, wall2)
        acceleration = agent.force / agent.mass
        agent.velocity += acceleration * constant.dt
        agent.position += agent.velocity * constant.dt
        # Save
        result.increment_simu_time(constant.dt)
        yield
