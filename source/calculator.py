# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def distance(v1, v2):
    d = v1 - v2
    return np.sqrt(np.dot(d, d))


# @numba.jit(nopython=True, cache=True)
def f_soc_ij(xi, xj, vi, vj, ri, rj):
    r"""
    Social interaction force between two agents `i` and `j`. [1]

    Params
    ------
    :param xi: Position (center of mass) of agent i.
    :param xj: Position (center of mass) of agent j.
    :param vi: Velocity of agent i.
    :param vj: Velocity of agent j.
    :param ri: Radius of agent i.
    :param rj: Radius of agent j.
    :param sight: If relative distance between two agents is greater than sight force is zero.
    :param force_max: Maximum magnitude of force. Forces greater than this are scaled to force max.
    :return: Vector of length 2 containing `x` and `y` components of force
             on agent i.

    References
    ----------
    [1] http://motion.cs.umn.edu/PowerLaw/
    """
    maxt = 999

    # Init output values.
    force = np.zeros(2)

    # Constants
    k = 1.5
    m = 2.0
    tau_0 = 3
    sight = 7
    force_max = 5

    dist = distance(xi, xj)
    if dist > sight:
        print("Not in sight")
        return force

    # Variables
    x = xj - xi  # position[i] - position[j]
    v = vi - vj  # velocity[i] - velocity[j]
    r = ri + rj  # radius + radius

    # If two agents are overlapping reduce r
    if r > dist:
        r = dist / 2.1

    b = np.dot(x, v)
    a = np.dot(v, v)
    c = np.dot(x, x) - r ** 2
    d = b ** 2 - a * c

    if (d < 0) or (- 0.001 < a < 0.001):
        return force

    d = np.sqrt(d)
    tau = (b - d) / a  # Time-to-collision

    if tau < 0 or tau > maxt:
        return force

    # Force is returned negative as repulsive force
    force -= k * np.exp(-tau / tau_0) * (m / tau + 1 / tau_0) * \
             (v - (v * b - x * a) / d) / (a * tau ** m)

    mag = np.sqrt(np.dot(force, force))
    if mag > force_max:
        # Scales magnitude of force to force max
        force *= force_max / mag

    return force


def f_soc_iw():
    force = np.zeros(2)
    return force


def f_c_ij():
    force = np.zeros(2)
    return force


def f_c_iw():
    force = np.zeros(2)
    return force


@numba.jit(nopython=True, cache=True)
def f_adjust(v_0, v, mass, tau):
    """

    :param v_0: Goal velocity of an agent
    :param v: Current velocity
    :param mass: Mass of an agent
    :param tau: Characteristic time where agent adapts its movement from current velocity to goal velocity
    :return: Vector of length 2 containing `x` and `y` components of force
             on agent i.
    """
    force = (v_0 - v) * mass / tau
    return force


def f_random_fluctuation():
    force = np.random.uniform(-1, 1, 2)
    return force


def f_tot(i, v_0, v, x, r, mass, tau):
    """
    :return: Vector of length 2 containing `x` and `y` components of force
             on agent i.
    """
    force = f_adjust(v_0, v[i], mass, tau) + f_random_fluctuation()
    for j in range(len(x)):
        if i == j:
            continue
        force += f_soc_ij(x[i], x[j], v[i], v[j], r[i], r[j])
    return force


def update_positions(x, v, gv, r, masses, tau, dt=0.01):
    """
    Updates positions and velocities of agents using forces affecting them with
    given timestep.

    Params
    ------
    :param dt:
    :return:
    """
    iteration = 0
    forces = np.zeros_like(v)
    while True:
        for i in range(len(x)):
            forces[i] = f_tot(i, gv[i], v, x, r, masses[i], tau)
        v += forces * dt
        x += v * dt
        iteration += 1
        # print(iteration)
        yield x
