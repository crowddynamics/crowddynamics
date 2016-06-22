import numba

from .models.three_circles import agent_agent_interaction, agent_wall_interaction


# TODO: Merge three circles and circular models


@numba.jit(nopython=True, nogil=True)
def agent_agent(agent):
    # n - 1 + n - 2 + ... + 1 = n^2 / 2
    for i in range(agent.size - 1):
        for j in range(i + 1, agent.size):
            agent_agent_interaction(i, j, agent)


@numba.jit(nopython=True, nogil=True)
def agent_wall(agent, wall):
    for w in range(wall.size):
        for i in range(agent.size):
            agent_wall_interaction(i, w, agent, wall)
