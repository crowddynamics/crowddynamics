import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def f_random_fluctuation(agent, f_max=1):
    force = agent.force
    for i in range(agent.size):
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(0, f_max)
        force[i][0] += magnitude * np.cos(angle)
        force[i][1] += magnitude * np.sin(angle)


@numba.jit(nopython=True, nogil=True)
def f_adjust(constant, agent):
    agent.force += (agent.mass / constant.tau_adj) * \
                   (agent.goal_velocity * agent.goal_direction - agent.velocity)
