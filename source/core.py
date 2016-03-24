# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time

import numpy

from source.social_force import f_tot


def update_positions(positions: numpy.ndarray,
                     velocities: numpy.ndarray,
                     goal_velocities: numpy.ndarray,
                     radii: numpy.ndarray,
                     masses: numpy.ndarray,
                     dt: float = 0.01) -> numpy.ndarray:
    """
    About
    -----
    Updates positions and velocities of agents using forces affecting them with
    given timestep. https://en.wikipedia.org/wiki/Euler_method

    Params
    ------
    :param positions:
    :param velocities:
    :param goal_velocities:
    :param radii:
    :param masses:
    :param dt:
    :return:
    """
    tau = 0.5  # [s] Characteristic time in which agent adjusts its movement
    tau_0 = 3.0  # [s] Max interaction range 2 - 4, aka interaction time horizon
    sight = 7.0  # [m] Max distance between agents for interaction to occur
    force_max = 5.0  # [N] Forces that are greater will be truncated to max force
    k = 1.2e5  # [kg/s^2]
    kappa = 2.4e5  # [kg/(m s)]
    a = 2e3  # [N]
    b = 0.08  # [m]

    t0 = time.clock()
    iteration = 0
    while True:
        forces = f_tot(goal_velocities, velocities, positions, radii, masses,
                       tau, tau_0, sight, force_max)
        velocities += forces * dt
        positions += velocities * dt
        iteration += 1
        t1 = time.clock()
        print(iteration, ':', t1 - t0)
        t0 = t1
        yield positions
