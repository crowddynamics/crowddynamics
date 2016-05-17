from timeit import default_timer as timer

import numba
import numpy as np

from source.core.force_adjust import f_adjust, f_random_fluctuation
from source.core.force_agents import f_ij
from source.core.force_walls import f_iw_tot


class MethodStats(object):
    def __init__(self):
        self._calls = []

    def _append(self, value):
        self._calls.append(value)

    def __str__(self):
        arr = np.array(self._calls)[1:]
        return "Number of calls: {} \n" \
               "Min call time: {} \n" \
               "Max call time: {} \n" \
               "Avg call time: {} \n" \
               "Variance: {} \n".format(arr.size+1, arr.min(), arr.max(),
                                        arr.mean(), arr.var())

    def __call__(self, f):
        def wrapper(*args, **kwargs):
            start = timer()
            ret = f(*args, **kwargs)
            end = timer()
            self._append(end - start)
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
    force = np.zeros(agent.shape)

    # Random Fluctuation
    force += f_random_fluctuation(agent)

    # Adjusting Force
    # TODO: Target direction updating algorithm
    force += f_adjust(constant, agent)

    # Agent - Agent Forces
    force += f_ij(constant, agent)

    # Agent - Wall Forces
    force += f_iw_tot(constant, agent, linear_wall)

    return force


@numba.jit(nopython=True, nogil=True)
def euler_method(constant, agent, linear_wall, dt):
    """
    Updates agent's velocity and position using euler method.

    Resources
    ---------
    - https://en.wikipedia.org/wiki/Euler_method
    """
    force = f_tot(constant, agent, linear_wall)
    acceleration = force / agent.mass
    agent.velocity += acceleration * dt
    agent.position += agent.velocity * dt


def system(constant, agent, linear_wall, dt=0.01):
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
    :param dt: Timestep
    :return:

    """
    # TODO: AOT complilation
    # TODO: Adaptive Euler method
    # TODO: Optional walls
    # TODO: Round walls
    # TODO: Gather stats
    stats = MethodStats()
    iteration = 0
    method = stats(euler_method)
    while True:
        method(constant, agent, linear_wall, dt)
        iteration += 1
        if iteration % 100 == 0:
            print(stats)
        yield agent
