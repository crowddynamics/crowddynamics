from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time

from source.core.social_force import acceleration


def update_positions(agents, constants, dt=0.01):
    """
    About
    -----
    Updates positions and velocities of agents using forces affecting them with
    given timestep.

    Params
    ------
    :param agents:
    :param constants:
    :param dt:
    :return:

    Resources
    ---------
    https://en.wikipedia.org/wiki/Euler_method
    """
    t0 = time.clock()
    iteration = 0
    while True:
        kwargs = dict(agents, **constants)
        acc = acceleration(**kwargs)
        agents['velocity'] += acc * dt
        agents['position'] += agents['velocity'] * dt

        t1 = time.clock()
        print(iteration, ':', round(t1 - t0, 4))
        t0 = t1

        iteration += 1
        yield agents
