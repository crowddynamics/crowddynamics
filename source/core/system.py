from timeit import default_timer as timer

import numba
import numpy as np

from source.core.force_adjust import f_adjust, f_random_fluctuation
from source.core.force_agent import f_agent_agent
from source.core.force_wall import f_agent_wall


class MethodStats(object):
    def __init__(self):
        self._calls = []

    def _append(self, value):
        self._calls.append(value)

    def dec_generator(self, gen):
        while True:
            start = timer()
            try:
                ret = next(gen)
            except GeneratorExit:
                return
            end = timer()
            self._append(end - start)
            if len(self._calls) % 100 == 0:
                print(self)
            yield ret

    def __str__(self):
        arr = np.array(self._calls)[1:]
        return "Number of calls: {} \n" \
               "Avg call time: {} \n".format(arr.size+1, arr.mean())

    def __call__(self, f):
        def wrapper(*args, **kwargs):
            start = timer()
            ret = f(*args, **kwargs)
            end = timer()
            self._append(end - start)
            return ret
        return wrapper


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
    f_adjust(constant, agent)            # _4.3 %
    f_agent_agent(constant, agent)       # 53.4 %
    f_agent_wall(constant, agent, wall)  # 33.7 %
    f_random_fluctuation(agent)          # _2.2 % of runtime


@numba.jit(nopython=True, nogil=True)
def euler_method(constant, agent, wall, dt):
    """
    Updates agent's velocity and position using euler method.

    Resources
    ---------
    - https://en.wikipedia.org/wiki/Euler_method
    """
    while True:
        # TODO: Target direction updating algorithm
        # agent.herding_behaviour()
        f_tot(constant, agent, wall)
        acceleration = agent.force / agent.mass
        agent.velocity += acceleration * dt
        agent.position += agent.velocity * dt
        agent.reset_force()
        yield


def system(constant, agent, wall):
    """
    About
    -----
    Updates positions and velocities of agents using forces affecting them with
    given timestep.

    Params
    ------
    :param wall:
    :param agent:
    :param constant:
    :param dt: Timestep
    :return:

    """
    # TODO: AOT complilation
    # TODO: Adaptive Euler method
    # TODO: Optional walls
    gen = euler_method(constant, agent, wall, dt=0.01)
    return MethodStats().dec_generator(gen)
