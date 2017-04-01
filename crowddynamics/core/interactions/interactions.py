"""Interactions

Interaction module has algorithms for computing the interactions between agents.
Interactions are computationally expensive, thus sophisticated algorithms are
required to efficiently compute them. Interactions are N-body problem.

Todo:
    - Rework walls
    - Toggleable helbing/power law
    - Neighborhood

"""
import numba
import numpy as np

from crowddynamics.core.interactions import distance_circle_circle, \
    distance_circle_line, distance_three_circle_line, distance_three_circle, \
    BlockList
from crowddynamics.core.motion import force_social_circular, \
    force_social_three_circle, force_social_linear_wall, force_contact
from crowddynamics.core.vector import rotate270, cross


@numba.jit(nopython=True, nogil=True)
def agent_agent_brute(agent, indices):
    r"""
    Interaction forces between set of agents.

    Computational complexity (number of iterations)

    .. math::
        n - 1 + n - 2 + ... + 1 =  \frac{(|N| - 1)^2}{2} \in \mathcal{O}(n^2)

    Args:
        agent (Agent):
        indices (numpy.ndarray): Subset of ``agent.indices``. If equal to ``agent.indices`` then
            brute force over all agents.

    """
    for l, i in enumerate(indices[:-1]):
        for j in indices[l + 1:]:
            if agent.three_circle:
                agent_agent_interaction_three_circle(i, j, agent)
            else:
                agent_agent_interaction_circle(i, j, agent)


@numba.jit(nopython=True, nogil=True)
def agent_agent_brute_disjoint(agent, indices_0, indices_1):
    r"""
    Interaction forces between two disjoint sets of agents. Assumes sets
    ``indices`` (:math:`S_{0}`) and ``indices2`` (:math:`S_{1}`) should be
    disjoint

    .. math::
       S_{0} \cap S_{1} = \emptyset

    Args:
        agent (Agent):
        indices_0 (numpy.ndarray):
        indices_1 (numpy.ndarray):

    """
    for i in indices_0:
        for j in indices_1:
            if agent.three_circle:
                agent_agent_interaction_three_circle(i, j, agent)
            else:
                agent_agent_interaction_circle(i, j, agent)


@numba.jit(nopython=True, nogil=True)
def agent_agent_block_list(agent):
    r"""Iteration over all agents using block list algorithm.

    Args:
        agent (Agent):

    """
    indices = agent.indices()
    blocks = BlockList(agent.position[indices], agent.sight_soc)
    n, m = blocks.shape

    # Neighbouring blocks
    nb = np.array(((1, 0), (1, 1), (0, 1), (1, -1)), dtype=np.int64)

    for i in range(n):
        for j in range(m):
            # Agents in the block
            ilist = blocks.get_block((i, j))
            indices_block = indices[ilist]

            # Forces between agents indices the block
            agent_agent_brute(agent, indices_block)

            # Forces between agent inside the block and neighbouring agents
            for k in range(len(nb)):
                i2, j2 = nb[k]
                if 0 <= (i + i2) < n and 0 <= (j + j2) < m:
                    ilist2 = blocks.get_block((i + i2, j + j2))
                    agent_agent_brute_disjoint(agent, indices_block, indices[ilist2])


@numba.jit(nopython=True, nogil=True)
def agent_wall(agent, wall):
    """
    Agent wall

    Args:
        agent:
        wall:

    """
    ind = agent.indices()
    for i in ind:
        for w in range(len(wall)):
            agent_obstacle_interaction_circle(i, w, agent, wall)


@numba.jit(nopython=True, nogil=True)
def agent_agent_interaction_circle(i, j, agent):
    """
    Interaction between two circular agents.

    Args:
        i:
        j:
        agent:

    """
    h, n = distance_circle_circle(agent.position[i], agent.radius[i],
                                  agent.position[j], agent.radius[j])
    if h < agent.sight_soc:
        force_i, force_j = force_social_circular(agent, i, j)

        if h < 0:
            t = rotate270(n)  # Tangent vector
            v = agent.velocity[i] - agent.velocity[j]  # Relative velocity
            force_i += force_contact(h, n, v, t, agent.mu[i], agent.kappa[i], agent.damping[i])
            force_j -= force_contact(h, n, v, t, agent.mu[j], agent.kappa[j], agent.damping[j])

        agent.force[i] += force_i
        agent.force[j] += force_j


@numba.jit(nopython=True, nogil=True)
def agent_agent_interaction_three_circle(i, j, agent):
    """
    Interaction between two three circle agents.

    Args:
        i:
        j:
        agent:

    Returns:

    """
    h, n, r_moment_i, r_moment_j = distance_three_circle(
        agent.positions(i), agent.radii(i),
        agent.positions(j), agent.radii(j)
    )
    if h < agent.sight_soc:
        force_i, force_j = force_social_three_circle(agent, i, j)

        if h < 0:
            t = rotate270(n)  # Tangent vector
            v = agent.velocity[i] - agent.velocity[j]  # Relative velocity
            force_i += force_contact(h, n, v, t, agent.mu[i], agent.kappa[i],
                                     agent.damping[i])
            force_j -= force_contact(h, n, v, t, agent.mu[j], agent.kappa[j],
                                     agent.damping[j])

        agent.force[i] += force_i
        agent.force[j] += force_j

        agent.torque[i] += cross(r_moment_i, force_i)
        agent.torque[j] += cross(r_moment_j, force_j)


@numba.jit(nopython=True, nogil=True)
def agent_obstacle_interaction_circle(i, w, agent, wall):
    """
    Interaction between circular agent and line obstacle.

    Args:
        i:
        w:
        agent:
        wall:

    """
    h, n = distance_circle_line(agent.position[i], agent.radius[i], wall[w])
    if h < agent.sight_wall:
        force = force_social_linear_wall(i, w, agent, wall)

        if h < 0:
            t = rotate270(n)  # Tangent
            v = agent.velocity[i]
            force += force_contact(h, n, v, t, agent.mu[i], agent.kappa[i],
                                   agent.damping[i])

        agent.force[i] += force


@numba.jit(nopython=True, nogil=True)
def agent_obstacle_interaction_three_circle(i, w, agent, wall):
    """
    Interaction between three circle agent and line obstacle.

    Args:
        i:
        w:
        agent:
        wall:

    """
    h, n, r_moment = distance_three_circle_line(
        agent.positions(i), agent.radii(i), wall[w]
    )
    if h < agent.sight_wall:
        force = force_social_linear_wall(i, w, agent, wall)

        if h < 0:
            t = rotate270(n)  # Tangent
            v = agent.velocity[i]
            force += force_contact(h, n, v, t, agent.mu[i], agent.kappa[i],
                                   agent.damping[i])

        agent.force[i] += force
        agent.torque[i] += cross(r_moment, force)
