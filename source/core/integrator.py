import numba

from source.core.force_adjust import f_adjust, f_random_fluctuation
from source.core.force_agent import f_agent_agent
from source.core.force_wall import f_agent_wall


@numba.jit(nopython=True, nogil=True)
def f_tot(constant, agent, wall):
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
        f_tot(constant, agent, wall)
        acceleration = agent.force / agent.mass
        agent.velocity += acceleration * constant.dt
        agent.position += agent.velocity * constant.dt
        agent.reset_force()
        # Save
        result.increment_simu_time(constant.dt)
        yield
