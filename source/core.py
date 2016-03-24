# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from source.calculator import f_tot


def update_positions(x, v, gv, r, mass, dt=0.01):
    """
    About
    -----
    Updates positions and velocities of agents using forces affecting them with
    given timestep.

    Params
    ------
    :param dt:
    :return:
    """
    tau = 0.5  # Characteristic time in which agent adjusts its movement
    tau_0 = 3.0  # Max interaction range 2 - 4, aka interaction time horizon
    sight = 7.0  # Max distance between agents for interaction to occur
    force_max = 5.0  # Forces that are greater will be truncated to max force

    import time
    t0 = time.clock()
    iteration = 0
    while True:
        forces = f_tot(gv, v, x, r, mass, tau, tau_0, sight, force_max)
        v += forces * dt
        x += v * dt
        iteration += 1
        t1 = time.clock()
        print(iteration, ':', t1-t0)
        t0 = t1
        yield x