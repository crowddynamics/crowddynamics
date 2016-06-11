import numba

from crowd_dynamics.core.force import force_adjust, force_random
from crowd_dynamics.core.interactions import agent_agent, agent_wall


# @numba.jit(nopython=True, nogil=True)
def explicit_euler_method(result, constant, agent, wall1=None, wall2=None):
    while True:
        # Target direction
        agent.goal_to_target_direction()

        # Forces
        agent.reset_force()
        agent.reset_force_debug()

        force_adjust(constant, agent)
        force_random(constant, agent)
        agent_agent(constant, agent)
        if wall1 is not None:
            agent_wall(constant, agent, wall1)
        if wall2 is not None:
            agent_wall(constant, agent, wall2)

        # Integration
        acceleration = agent.force / agent.mass
        agent.velocity += acceleration * constant.dt
        agent.position += agent.velocity * constant.dt

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
        agent.reset_force()
        agent.reset_force_debug()
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
        agent.reset_force()
        agent.reset_force_debug()
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
        agent.reset_force()
        agent.reset_force_debug()
        _f_tot2(constant, agent, wall1, wall2)
        acceleration = agent.force / agent.mass
        agent.velocity += acceleration * constant.dt
        agent.position += agent.velocity * constant.dt
        # Save
        result.increment_simu_time(constant.dt)
        yield
