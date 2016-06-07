import numba

from src.core.force import force_adjust, force_random_fluctuation
from src.core.force_agent import f_agent_agent
from src.core.force_wall import f_agent_wall


@numba.jit(nopython=True, nogil=True)
def _f_tot0(constant, agent):
    f_agent_agent(constant, agent)
    force_adjust(constant, agent)
    force_random_fluctuation(constant, agent)


@numba.jit(nopython=True, nogil=True)
def euler_method0(result, constant, agent):
    while True:
        # Target direction
        agent.goal_to_target_direction()
        # Update  position
        agent.reset_force()
        _f_tot0(constant, agent)
        acceleration = agent.force / agent.mass
        agent.velocity += acceleration * constant.dt
        agent.position += agent.velocity * constant.dt
        # Save
        result.increment_simu_time(constant.dt)
        yield


@numba.jit(nopython=True, nogil=True)
def _f_tot(constant, agent, wall):
    f_agent_agent(constant, agent)        # 53.4 %
    f_agent_wall(constant, agent, wall)   # 33.7 %
    force_adjust(constant, agent)              # 4.3 %
    force_random_fluctuation(constant, agent)  # 2.2 %


@numba.jit(nopython=True, nogil=True)
def euler_method(result, constant, agent, wall):
    while True:
        # Target direction
        agent.goal_to_target_direction()
        # Update  position
        agent.reset_force()
        _f_tot(constant, agent, wall)
        acceleration = agent.force / agent.mass
        agent.velocity += acceleration * constant.dt
        agent.position += agent.velocity * constant.dt
        # Save
        result.increment_simu_time(constant.dt)
        yield


@numba.jit(nopython=True, nogil=True)
def _f_tot2(constant, agent, wall1, wall2):
    f_agent_agent(constant, agent)
    f_agent_wall(constant, agent, wall1)
    f_agent_wall(constant, agent, wall2)
    force_adjust(constant, agent)
    force_random_fluctuation(constant, agent)


@numba.jit(nopython=True, nogil=True)
def euler_method2(result, constant, agent, wall1, wall2):
    while True:
        # Target direction
        agent.goal_to_target_direction()
        # Update  position
        agent.reset_force()
        _f_tot2(constant, agent, wall1, wall2)
        acceleration = agent.force / agent.mass
        agent.velocity += acceleration * constant.dt
        agent.position += agent.velocity * constant.dt
        # Save
        result.increment_simu_time(constant.dt)
        yield
