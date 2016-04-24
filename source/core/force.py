import numba
import numpy as np

from source.core.force_agents import f_ij
from source.core.force_walls import f_iw_linear_tot


@numba.jit(nopython=True, nogil=True)
def f_random_fluctuation():
    """

    :return: Uniformly distributed random force.
    """
    # TODO: np.random.seed
    # for loop so compilation can be done with numba
    force = np.zeros(2)
    for i in range(len(force)):
        force[i] = np.random.uniform(-1, 1)
    return force


@numba.jit(nopython=True, nogil=True)
def f_adjust_i(v_0, e_i, v_i, m_i, tau_i):
    """
    Params
    ------
    :param v_0: Goal velocity of an agent
    :param e_i:
    :param v_i: Current velocity
    :param m_i: Mass of an agent
    :param tau_i: Characteristic time where agent adapts its movement from current velocity to goal velocity
    :return: Vector of length 2 containing `x` and `y` components of force on agent i.
    """
    force = (m_i / tau_i) * (v_0 * e_i - v_i)
    return force


@numba.jit(nopython=True, nogil=True)
def f_tot_i(i, v_0, e_i, v, x, r, mass, linear_wall,
            tau_adj, tau_0, k, sight, f_max, mu, kappa, a, b):
    """
    Total force on individual agent i.

    :return: Vector of length 2 containing `x` and `y` components of force on agent i.
    """

    # Initialize
    force = np.zeros(2)

    # Adjusting Force
    # TODO: Target direction updating algorithm
    force += f_adjust_i(v_0, e_i, v[i], mass, tau_adj)

    # Agent - Agent Forces
    force += f_ij(i, x, v, r, k, tau_0, sight, f_max, mu, kappa)

    # Agent - Wall Forces
    force += f_iw_linear_tot(i, x, v, r, linear_wall, f_max, sight, mu, kappa,
                             a, b)

    # Random Fluctuation
    force += f_random_fluctuation()

    return force

