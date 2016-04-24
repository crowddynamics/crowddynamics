from timeit import default_timer as timer

import numba
import numpy as np

from source.core.force import f_tot_i


def timeit(f):
    def wrapper(*args, **kwargs):
        start = timer()
        ret = f(*args, **kwargs)
        end = timer()
        print('Wall time:', round(end - start, 4))
        return ret
    return wrapper


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


def system(agents, walls, constants, t_delta):
    """
    About
    -----
    Updates positions and velocities of agents using forces affecting them with
    given timestep.

    Params
    ------
    :param agents:
    :param constants:
    :param t_delta: Timestep
    :return:

    Resources
    ---------
    - https://en.wikipedia.org/wiki/Euler_method
    """
    # round_wall = walls['round_wall']
    # linear_wall = walls['linear_wall']

    @timeit
    def update(i):
        kwargs = dict(agents, **constants)
        # kwargs = dict(kwargs, **walls)
        kwargs['linear_wall'] = walls['linear_wall']
        acc = acceleration(**kwargs)
        agents['velocity'] += acc * t_delta
        agents['position'] += agents['velocity'] * t_delta
        print('Simulation Time:', round(i * t_delta, 4), end=' ')

    iteration = 0
    while True:
        update(iteration)
        iteration += 1
        yield agents
