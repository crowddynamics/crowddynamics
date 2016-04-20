import numba
import numpy as np

from source.core.force_agents import f_ij
from source.core.force_walls import f_iw_linear_tot


@numba.jit(nopython=True, nogil=True)
def f_random_fluctuation():
    """

    :return: Uniformly distributed random force.
    """
    # TODO: seed
    force = np.zeros(2)
    for i in range(len(force)):
        # for loop so compilation can be done with numba
        force[i] = np.random.uniform(-1, 1)
    return force


# @numba.jit(nopython=True, nogil=True)
def update_goal_direction(e_i, p_i):
    """
    Update goal direction.
    """
    pass


@numba.jit(nopython=True, nogil=True)
def f_adjust_i(v_0, e_i, v_i, mass_i, tau_i):
    """
    Params
    ------
    :param v_0: Goal velocity of an agent
    :param e_i:
    :param v_i: Current velocity
    :param mass_i: Mass of an agent
    :param tau_i: Characteristic time where agent adapts its movement from current velocity to goal velocity
    :return: Vector of length 2 containing `x` and `y` components of force on agent i.
    """
    # TODO: v_0 = v_0 * e_i
    force = (mass_i / tau_i) * (v_0 * e_i - v_i)
    return force


@numba.jit(nopython=True, nogil=True)
def f_tot_i(i, v_0, e_i, v, x, r, mass, linear_wall,
            tau_adj, tau_0, k, sight, f_max, mu, kappa, a, b):
    """
    Total force on individual agent i.

    :return: Vector of length 2 containing `x` and `y` components of force
             on agent i.
    """

    # Initialize
    force = np.zeros(2)

    # Movement adjusting force
    force += f_adjust_i(v_0, e_i, v[i], mass, tau_adj)

    # Random Fluctuation
    force += f_random_fluctuation()

    # Agent-Agent
    force += f_ij(i, x, v, r, k, tau_0, sight, f_max, mu, kappa)

    # Agent-Wall
    force += f_iw_linear_tot(i, x, v, r, linear_wall, f_max, sight, mu, kappa,
                             a, b)

    return force


@numba.jit(nopython=True, nogil=True)
def acceleration(goal_velocity, goal_direction, velocity, position, radius,
                 mass, linear_wall, tau_adj, k, tau_0, sight, f_max,
                 mu, kappa, a, b):
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
    # TODO: AOT complilation
    # TODO: Adaptive Euler Method
    # TODO: Mass & radius -> scalar & vector inputs
    acc = np.zeros_like(velocity)
    for i in range(len(position)):
        f = f_tot_i(i, goal_velocity, goal_direction[i], velocity, position,
                    radius, mass[i], linear_wall, tau_adj, k, tau_0, sight,
                    f_max, mu, kappa, a, b)
        acc[i] = f / mass[i]
    return acc
