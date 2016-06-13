import numba

from .circular_model import agent_agent_interaction, agent_wall_interaction


@numba.jit(nopython=True, nogil=True)
def agent_agent(constant, agent):
    # n - 1 + n - 2 + ... + 1 = n^2 / 2
    for i in range(agent.size - 1):
        for j in range(i + 1, agent.size):
            # Function params
            agent_agent_interaction(i, j, constant, agent)


@numba.jit(nopython=True, nogil=True)
def agent_wall(constant, agent, wall):
    for w in range(wall.size):
        for i in range(agent.size):
            agent_wall_interaction(i, w, constant, agent, wall)
