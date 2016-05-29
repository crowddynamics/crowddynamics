import numba

from src.core.force_adjust import f_adjust, f_random_fluctuation
from src.core.force_agent import f_agent_agent
from src.core.force_wall import f_agent_wall


@numba.jit(nopython=True, nogil=True)
def _f_tot0(constant, agent):
    """
    About
    -----
    Total forces on all agents in the system. Uses `Helbing's` social force model
    [1] and with power law [2].

    Params
    ------
    :return: Array of forces.

    References
    ----------
    [1] http://www.nature.com/nature/journal/v407/n6803/full/407487a0.html \n
    [2] http://motion.cs.umn.edu/PowerLaw/
    """
    f_agent_agent(constant, agent)
    f_adjust(constant, agent)
    f_random_fluctuation(constant, agent)


@numba.jit(nopython=True, nogil=True)
def euler_method0(result, constant, agent):
    """
    Updates agent's velocity and position using euler method.

    Resources
    ---------
    - https://en.wikipedia.org/wiki/Euler_method
    """
    while True:
        # Target direction
        agent.goal_to_target_direction()
        # Update  position
        _f_tot0(constant, agent)
        acceleration = agent.force / agent.mass
        agent.velocity += acceleration * constant.dt
        agent.position += agent.velocity * constant.dt
        agent.reset_force()
        # Save
        result.increment_simu_time(constant.dt)
        yield


@numba.jit(nopython=True, nogil=True)
def _f_tot(constant, agent, wall):
    """
    About
    -----
    Total forces on all agents in the system. Uses `Helbing's` social force model
    [1] and with power law [2].

    Params
    ------
    :return: Array of forces.

    References
    ----------
    [1] http://www.nature.com/nature/journal/v407/n6803/full/407487a0.html \n
    [2] http://motion.cs.umn.edu/PowerLaw/
    """
    f_agent_agent(constant, agent)        # 53.4 %
    f_agent_wall(constant, agent, wall)   # 33.7 %
    f_adjust(constant, agent)              # 4.3 %
    f_random_fluctuation(constant, agent)  # 2.2 %


@numba.jit(nopython=True, nogil=True)
def euler_method(result, constant, agent, wall):
    """
    Updates agent's velocity and position using euler method.

    Resources
    ---------
    - https://en.wikipedia.org/wiki/Euler_method
    """
    while True:
        # Target direction
        agent.goal_to_target_direction()
        # Update  position
        _f_tot(constant, agent, wall)
        acceleration = agent.force / agent.mass
        agent.velocity += acceleration * constant.dt
        agent.position += agent.velocity * constant.dt
        agent.reset_force()
        # Save
        result.increment_simu_time(constant.dt)
        yield


@numba.jit(nopython=True, nogil=True)
def _f_tot2(constant, agent, wall1, wall2):
    """
    About
    -----
    Total forces on all agents in the system. Uses `Helbing's` social force model
    [1] and with power law [2].

    Params
    ------
    :return: Array of forces.

    References
    ----------
    [1] http://www.nature.com/nature/journal/v407/n6803/full/407487a0.html \n
    [2] http://motion.cs.umn.edu/PowerLaw/
    """
    f_agent_agent(constant, agent)
    f_agent_wall(constant, agent, wall1)
    f_agent_wall(constant, agent, wall2)
    f_adjust(constant, agent)
    f_random_fluctuation(constant, agent)


@numba.jit(nopython=True, nogil=True)
def euler_method2(result, constant, agent, wall1, wall2):
    """
    Updates agent's velocity and position using euler method.

    Resources
    ---------
    - https://en.wikipedia.org/wiki/Euler_method
    """
    while True:
        # Target direction
        agent.goal_to_target_direction()
        # Update  position
        _f_tot2(constant, agent, wall1, wall2)
        acceleration = agent.force / agent.mass
        agent.velocity += acceleration * constant.dt
        agent.position += agent.velocity * constant.dt
        agent.reset_force()
        # Save
        result.increment_simu_time(constant.dt)
        yield
