import numba
import numpy as np


@numba.jit( nopython=True, nogil=True)
def f_random_fluctuation(size, f_max=1):
    """

    :param size:
    :param f_max: Maximum magnitude of the force
    :return:
    :return: Uniformly distributed random force.
    """
    force = np.zeros((size, 2))
    for i in range(size):
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(0, f_max)
        force[i][0] = magnitude * np.cos(angle)
        force[i][1] = magnitude * np.sin(angle)
    return force


@numba.jit(nopython=True, nogil=True)
def f_adjust(constant, agent):
    force = (agent.mass / constant.tau_adj) * \
            (agent.goal_velocity * agent.goal_direction - agent.velocity)
    return force
