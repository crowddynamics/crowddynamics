from timeit import default_timer as timer

import numba
import numpy as np

from source.core.force_adjust import f_adjust_i, f_random_fluctuation
from source.core.force_agents import f_ij
from source.core.force_walls import f_iw_linear_tot


def timeit(f):
    def wrapper(*args, **kwargs):
        start = timer()
        ret = f(*args, **kwargs)
        end = timer()
        print('Wall time:', round(end - start, 4))
        return ret
    return wrapper


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
    acc = np.zeros_like(velocity)
    for i in range(len(position)):
        f = f_tot_i(i, goal_velocity, goal_direction[i], velocity, position,
                    radius, mass[i], linear_wall, tau_adj, k, tau_0, sight,
                    f_max, mu, kappa, a, b)
        acc[i] = f / mass[i]
    return acc


@numba.jit(nopython=True, nogil=True)
def euler_method(agent, force, dt):
    """
    Updates agent's velocity and position using euler method.

    Resources
    ---------
    - https://en.wikipedia.org/wiki/Euler_method
    """
    acc = force / agent.mass
    agent.velocity += acc * dt
    agent.position += agent.velocity * dt


def system(agent, wall, constant, t_delta):
    """
    About
    -----
    Updates positions and velocities of agents using forces affecting them with
    given timestep.

    Params
    ------
    :param agent:
    :param constant:
    :param t_delta: Timestep
    :return:

    """
    # TODO: AOT complilation
    # TODO: Adaptive Euler method

    @timeit
    def update(i):
        kwargs = dict(agent, **constant)
        kwargs['linear_wall'] = wall['linear_wall']
        acc = acceleration(**kwargs)
        agent['velocity'] += acc * t_delta
        agent['position'] += agent['velocity'] * t_delta

        print('Simulation Time:', round(i * t_delta, 4), end=' ')

    iteration = 0
    while True:
        update(iteration)
        iteration += 1
        yield agent
