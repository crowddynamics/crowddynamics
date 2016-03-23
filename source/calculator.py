# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numba
import numpy as np
from numba import float64


@numba.jit("float64[:](float64[:], float64[:], "
           "float64[:], float64[:], float64, float64)",
           nopython=True)
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
    # Init output values.
    force = np.zeros(2)

    # Constants
    k = 1.5  # Constant
    m = 2.0  # Exponent in power law
    tau_0 = 3.0  # Max interaction range 2 - 4, aka interaction time horizon
    sight = 7.0  # Max distance between agents for interaction to occur
    force_max = 5.0  # Forces that are greater will be truncated to max force
    maxt = 999.0

    # Variables
    x = xj - xi  # position[j] - position[i]
    v = vi - vj  # velocity[i] - velocity[j]
    r = ri + rj  # radius[i] + radius[j]

    dot_x = np.dot(x, x)
    dist = np.sqrt(dot_x)
    # No force if another agent is not in range of sight
    if dist > sight:
        return force

    # If two agents are overlapping reduce r
    if r > dist:
        r = 0.50 * dist

    b = np.dot(x, v)
    a = np.dot(v, v)
    c = dot_x - r ** 2
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


@numba.jit(nopython=True)
def f_soc_ij_tot(i, x, v, r):
    force = np.zeros(2)
    for j in range(len(x)):
        if i == j:
            continue
        force += f_soc_ij(x[i], x[j], v[i], v[j], r[i], r[j])
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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
def f_random_fluctuation():
    force = np.zeros(2)
    for i in range(len(force)):
        force[i] = np.random.uniform(-1, 1)
    return force


@numba.jit(nopython=True)
def f_tot_i(i, v_0, v, x, r, mass, tau):
    """
    :return: Vector of length 2 containing `x` and `y` components of force
             on agent i.
    """
    force = f_adjust(v_0, v[i], mass, tau) + \
            f_soc_ij_tot(i, x, v, r) + \
            f_random_fluctuation()
    return force


@numba.jit(nopython=True)
def f_tot(gv, v, x, r, masses, tau):
    forces = np.zeros_like(v)
    for i in range(len(x)):
            forces[i] = f_tot_i(i, gv[i], v, x, r, masses[i], tau)
    return forces


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
    while True:
        forces = f_tot(gv, v, x, r, masses, tau)
        v += forces * dt
        x += v * dt
        iteration += 1
        print(iteration)
        yield x
