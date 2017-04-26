r"""Interactions

Interaction module has algorithms for computing total forces affecting the
agents. This is essentially an N-body problem.

- Brute Force

  Number of iterations for set of :math:`N > 0` agents  

  .. math::
      n - 1 + n - 2 + ... + 1 =  \frac{(N - 1)^2}{2} \in \mathcal{O}(N^2)
    
  Number of iterations for two disjoint sets of :math:`N > 0` and :math:`M > 0` 
  agents
  
  .. math::
     M N

- Block List

  Number of iterations if maximum number of agents that can be fit into a cell 
  is :math:`M` is some constant.
  
  Iterations per block
  
  .. math::
     I = \frac{(M - 1)^2}{2} + \frac{9}{2} * M^{2}
     
  For :math:`N` agents the number of blocks :math:`N / M`. 
  
  .. math::
     I \frac{N}{M} = \frac{N}{M} \left(5 M^{2} - M + \frac{1}{2}\right) \in 
     \mathcal{O}(N)

"""

import numba
import numpy as np
from numba import void, i8, typeof

from crowddynamics.core.interactions.distance import distance_circles, \
    distance_circle_line, distance_three_circle_line, distance_three_circles
from crowddynamics.core.interactions.partitioning import block_list, \
    get_block
from crowddynamics.core.motion.contact import force_contact
from crowddynamics.core.motion.power_law import \
    force_social_circular, force_social_three_circle
from crowddynamics.core.structures.agents import agent_type_circular, \
    agent_type_three_circle, is_model
from crowddynamics.core.structures.obstacles import obstacle_type_linear
from crowddynamics.core.vector import rotate270, cross


# TODO: load from config
# Reach of the social force.
SIGTH_SOC = 3.0


# Individual interactions


@numba.jit(void(i8, i8, typeof(agent_type_circular)[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_interaction_circle(i, j, agent):
    """Interaction between two circular agents."""
    h, n = distance_circles(agent[i]['position'], agent[i]['radius'],
                            agent[j]['position'], agent[j]['radius'])

    if h < SIGTH_SOC:
        force_i, force_j = force_social_circular(agent, i, j)

        if h < 0:
            t = rotate270(n)  # Tangent vector
            v = agent[i]['velocity'] - agent[j]['velocity']  # Relative velocity
            force_i += force_contact(h, n, v, t, agent[i]['mu'], agent[i]['kappa'], agent[i]['damping'])
            force_j -= force_contact(h, n, v, t, agent[j]['mu'], agent[j]['kappa'], agent[j]['damping'])

        agent[i]['force'][:] += force_i
        agent[j]['force'][:] += force_j


@numba.jit(void(i8, i8, typeof(agent_type_three_circle)[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_interaction_three_circle(i, j, agent):
    """Interaction between two three circle agents."""
    # Positions: center, left, right
    x_i = (agent[i]['position'], agent[i]['position_ls'], agent[i]['position_rs'])
    x_j = (agent[j]['position'], agent[j]['position_ls'], agent[j]['position_rs'])

    # Radii of torso and shoulders
    r_i = (agent[i]['r_t'], agent[i]['r_s'], agent[i]['r_s'])
    r_j = (agent[j]['r_t'], agent[j]['r_s'], agent[j]['r_s'])

    h, n, r_moment_i, r_moment_j = distance_three_circles(x_i, r_i, x_j, r_j)

    if h < SIGTH_SOC:
        force_i, force_j = force_social_three_circle(agent, i, j)

        if h < 0:
            t = rotate270(n)  # Tangent vector
            v = agent[i]['velocity'] - agent[j]['velocity']  # Relative velocity
            force_i += force_contact(h, n, v, t, agent[i]['mu'], agent[i]['kappa'], agent[i]['damping'])
            force_j -= force_contact(h, n, v, t, agent[j]['mu'], agent[j]['kappa'], agent[j]['damping'])

        agent[i]['force'][:] += force_i
        agent[j]['force'][:] += force_j

        agent[i]['torque'] += cross(r_moment_i, force_i)
        agent[j]['torque'] += cross(r_moment_j, force_j)


@numba.jit(void(i8, i8, typeof(agent_type_circular)[:],
                typeof(obstacle_type_linear)[:]),
           nopython=True, nogil=True, cache=True)
def agent_obstacle_interaction_circle(i, w, agent, wall):
    """Interaction between circular agent and line obstacle."""
    h, n = distance_circle_line(agent[i]['position'], agent[i]['radius'],
                                wall[w]['p0'], wall[w]['p1'])
    if h < 0:
        t = rotate270(n)  # Tangent
        v = agent[i]['velocity']
        force = force_contact(h, n, v, t, agent[i]['mu'],
                              agent[i]['kappa'], agent[i]['damping'])

        agent[i]['force'][:] += force


@numba.jit(void(i8, i8, typeof(agent_type_three_circle)[:],
                typeof(obstacle_type_linear)[:]),
           nopython=True, nogil=True, cache=True)
def agent_obstacle_interaction_three_circle(i, w, agent, wall):
    """Interaction between three circle agent and line obstacle."""
    x_i = (agent[i]['position'], agent[i]['position_ls'], agent[i]['position_rs'])
    r_i = (agent[i]['r_t'], agent[i]['r_s'], agent[i]['r_s'])

    h, n, r_moment = distance_three_circle_line(x_i, r_i,
                                                wall[w]['p0'], wall[w]['p1'])
    if h < 0:
        t = rotate270(n)  # Tangent
        v = agent[i]['velocity']
        force = force_contact(h, n, v, t, agent[i]['mu'],
                              agent[i]['kappa'], agent[i]['damping'])

        agent[i]['force'][:] += force
        agent[i]['torque'] += cross(r_moment, force)


# Full interactions


@numba.jit(void(typeof(agent_type_circular)[:], i8[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_brute_circular(agent, indices):
    """Compute interaction forces between set of agents using brute force."""
    for l, i in enumerate(indices[:-1]):
        for j in indices[l + 1:]:
            agent_agent_interaction_circle(i, j, agent)


@numba.jit(void(typeof(agent_type_three_circle)[:], i8[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_brute_three_circle(agent, indices):
    """Compute interaction forces between set of agents using brute force."""
    for l, i in enumerate(indices[:-1]):
        for j in indices[l + 1:]:
            agent_agent_interaction_three_circle(i, j, agent)


@numba.jit(void(typeof(agent_type_circular)[:], i8[:], i8[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_brute_disjoint_circular(agent, indices_0, indices_1):
    """Compute interaction forces between two disjoint sets and of agents using
    brute force. Disjoint sets ``indices_0 ∩ indices_1 = ∅``."""
    for i in indices_0:
        for j in indices_1:
            agent_agent_interaction_circle(i, j, agent)


@numba.jit(void(typeof(agent_type_three_circle)[:], i8[:], i8[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_brute_disjoint_three_circle(agent, indices_0, indices_1):
    """Compute interaction forces between two disjoint sets and of agents using
    brute force. Disjoint sets ``indices_0 ∩ indices_1 = ∅``."""
    for i in indices_0:
        for j in indices_1:
            agent_agent_interaction_three_circle(i, j, agent)


@numba.jit(void(typeof(agent_type_circular)[:],
                i8[:], i8[:], i8[:], i8[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_block_list_circular(agent, index_list, count, offset, shape):
    """Iteration over all agents using block list algorithm."""
    # Neighbouring blocks
    nb = np.array(((1, 0), (1, 1), (0, 1), (1, -1)), dtype=np.int64)
    n, m = shape
    for i in range(n):
        for j in range(m):
            # Forces between agents inside the block
            ilist = get_block((i, j), index_list, count, offset, shape)
            agent_agent_brute_circular(agent, ilist)

            # Forces between agent inside the block and neighbouring agents
            for k in range(len(nb)):
                i2, j2 = nb[k]
                if 0 <= (i + i2) < n and 0 <= (j + j2) < m:
                    ilist2 = get_block((i + i2, j + j2), index_list, count,
                                       offset, shape)
                    agent_agent_brute_disjoint_circular(agent, ilist, ilist2)


@numba.jit(void(typeof(agent_type_three_circle)[:],
                i8[:], i8[:], i8[:], i8[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_block_list_three_circle(agent, index_list, count, offset, shape):
    """Iteration over all agents using block list algorithm."""
    # Neighbouring blocks
    nb = np.array(((1, 0), (1, 1), (0, 1), (1, -1)), dtype=np.int64)
    n, m = shape
    for i in range(n):
        for j in range(m):
            # Forces between agents inside the block
            ilist = get_block((i, j), index_list, count, offset, shape)
            agent_agent_brute_three_circle(agent, ilist)

            # Forces between agent inside the block and neighbouring agents
            for k in range(len(nb)):
                i2, j2 = nb[k]
                if 0 <= (i + i2) < n and 0 <= (j + j2) < m:
                    ilist2 = get_block((i + i2, j + j2), index_list, count,
                                       offset, shape)
                    agent_agent_brute_disjoint_three_circle(agent,
                                                            ilist,
                                                            ilist2)


def agent_agent_block_list(agent):
    index_list, count, offset, shape = block_list(agent['position'], SIGTH_SOC)
    if is_model(agent, 'circular'):
        agent_agent_block_list_circular(agent, index_list, count, offset, shape)
    elif is_model(agent, 'three_circle'):
        agent_agent_block_list_three_circle(agent, index_list, count, offset, shape)


def agent_agent_block_list_multithreaded(agent):
    pass


@numba.jit(void(typeof(agent_type_circular)[:], typeof(obstacle_type_linear)[:]),
           nopython=True, nogil=True, cache=True)
def circular_agent_linear_wall(agent, wall):
    """Agent wall"""
    for i in range(len(agent)):
        for w in range(len(wall)):
            agent_obstacle_interaction_circle(i, w, agent, wall)


@numba.jit(void(typeof(agent_type_three_circle)[:], typeof(obstacle_type_linear)[:]),
           nopython=True, nogil=True, cache=True)
def three_circle_agent_linear_wall(agent, wall):
    """Agent wall"""
    for i in range(len(agent)):
        for w in range(len(wall)):
            agent_obstacle_interaction_three_circle(i, w, agent, wall)
