import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def f_random_fluctuation(f_max=1):
    """

    :param f_max: Maximum magnitude of the force
    :return:
    :return: Uniformly distributed random force.
    """
    # TODO: np.random.seed
    force = np.zeros(2)
    angle = np.random.uniform(0, 2 * np.pi)
    magnitude = np.random.uniform(0, f_max)
    force[0] = magnitude * np.cos(angle)
    force[1] = magnitude * np.sin(angle)
    return force


@numba.jit(nopython=True, nogil=True)
def f_adjust_i(v_0, e_i, v_i, m_i, tau_i):
    """

    :param v_0: Goal velocity of an agent
    :param e_i:
    :param v_i: Current velocity
    :param m_i: Mass of an agent
    :param tau_i: Characteristic time where agent adapts its movement from current velocity to goal velocity
    :return: Vector of length 2 containing `x` and `y` components of force on agent i.
    """
    force = (m_i / tau_i) * (v_0 * e_i - v_i)
    return force
