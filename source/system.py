from timeit import default_timer as timer

import numpy as np

from source.core.integrator import euler_method


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