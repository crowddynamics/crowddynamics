from timeit import default_timer as timer

import numba
import numpy as np
from sympy.abc import kappa

from source.core.force_adjust import f_adjust, f_random_fluctuation
from source.core.force_agents import f_ij
from source.core.force_walls import f_iw_linear_tot


def timeit(f):
    def wrapper(*args, **kwargs):
        start = timer()
        ret = f(*args, **kwargs)
        end = timer()
        print('Wall time:', round(end - start, 7))
        return ret

    return wrapper


@numba.jit(nopython=True, nogil=True)
def f_tot(constant, agent, linear_wall):
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
    force = np.zeros((agent.size, 2))

    # Random Fluctuation
    force += f_random_fluctuation(agent.size)

    # Adjusting Force
    # TODO: Target direction updating algorithm
    force += f_adjust(constant, agent)

    # Agent - Agent Forces
    force += f_ij(constant, agent)

    # Agent - Wall Forces
    force += f_iw_linear_tot(constant, agent, linear_wall)

    return force


@numba.jit(nopython=True, nogil=True)
def euler_method(agent, force, dt):
    """
    Updates agent's velocity and position using euler method.

    Resources
    ---------
    - https://en.wikipedia.org/wiki/Euler_method
    """
    acc = force / agent.mass
    agent.velocity += acc * dt
    agent.position += agent.velocity * dt


def system(constant, agent, linear_wall, t_delta=0.01):
    """
    About
    -----
    Updates positions and velocities of agents using forces affecting them with
    given timestep.

    Params
    ------
    :param linear_wall:
    :param agent:
    :param constant:
    :param t_delta: Timestep
    :return:

    """

    # TODO: AOT complilation
    # TODO: Adaptive Euler method

    @timeit
    def update(i):
        forces = f_tot(constant, agent, linear_wall)
        euler_method(agent, forces, t_delta)
        print('Simulation Time:', round(i * t_delta, 4), end=' ')

    iteration = 0
    while True:
        update(iteration)
        iteration += 1
        yield agent
