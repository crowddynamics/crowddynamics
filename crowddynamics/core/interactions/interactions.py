r"""
Interactions
------------

.. math::
   \mathbf{f}_{i}^{agent-agent} = 
   \sum_{j\neq i}^{} \left(\mathbf{f}_{ij}^{soc} + \mathbf{f}_{ij}^{c}\right)

.. math::
   \mathbf{f}_{i}^{agent-obstacles} = \sum_{w}^{} \mathbf{f}_{iw}^{c}

.. math::
   M_{i}^{agent-agent} = 
   \sum_{j\neq i}^{} \left(M_{ij}^{soc} + M_{ij}^{c}\right)

.. math::
   M_{i}^{agent-obstacles} = \sum_{w}^{} M_{iw}^{c}

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
from crowddynamics.core.vector2D import rotate270, cross


# TODO: load from config
# Reach of the social force.
SIGTH_SOC = 3.0


# Individual interactions


@numba.jit(void(i8, i8, typeof(agent_type_circular)[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_interaction_circle(i, j, agents):
    """Interaction between two circular agents."""
    h, n = distance_circles(agents[i]['position'], agents[i]['radius'],
                            agents[j]['position'], agents[j]['radius'])

    if h < SIGTH_SOC:
        force_i, force_j = force_social_circular(agents, i, j)

        if h < 0:
            t = rotate270(n)  # Tangent vector
            v = agents[i]['velocity'] - agents[j]['velocity']  # Relative velocity
            force_i += force_contact(h, n, v, t, agents[i]['mu'], agents[i]['kappa'], agents[i]['damping'])
            force_j -= force_contact(h, n, v, t, agents[j]['mu'], agents[j]['kappa'], agents[j]['damping'])

        agents[i]['force'][:] += force_i
        agents[j]['force'][:] += force_j


@numba.jit(void(i8, i8, typeof(agent_type_three_circle)[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_interaction_three_circle(i, j, agents):
    """Interaction between two three circle agents."""
    # Positions: center, left, right
    x_i = (agents[i]['position'], agents[i]['position_ls'], agents[i]['position_rs'])
    x_j = (agents[j]['position'], agents[j]['position_ls'], agents[j]['position_rs'])

    # Radii of torso and shoulders
    r_i = (agents[i]['r_t'], agents[i]['r_s'], agents[i]['r_s'])
    r_j = (agents[j]['r_t'], agents[j]['r_s'], agents[j]['r_s'])

    h, n, r_moment_i, r_moment_j = distance_three_circles(x_i, r_i, x_j, r_j)

    if h < SIGTH_SOC:
        force_i, force_j = force_social_three_circle(agents, i, j)

        if h < 0:
            t = rotate270(n)  # Tangent vector
            v = agents[i]['velocity'] - agents[j]['velocity']  # Relative velocity
            force_i += force_contact(h, n, v, t, agents[i]['mu'], agents[i]['kappa'], agents[i]['damping'])
            force_j -= force_contact(h, n, v, t, agents[j]['mu'], agents[j]['kappa'], agents[j]['damping'])

        agents[i]['force'][:] += force_i
        agents[j]['force'][:] += force_j

        agents[i]['torque'] += cross(r_moment_i, force_i)
        agents[j]['torque'] += cross(r_moment_j, force_j)


@numba.jit(void(i8, i8, typeof(agent_type_circular)[:],
                typeof(obstacle_type_linear)[:]),
           nopython=True, nogil=True, cache=True)
def agent_obstacle_interaction_circle(i, w, agents, obstacles):
    """Interaction between circular agent and line obstacle."""
    h, n = distance_circle_line(agents[i]['position'], agents[i]['radius'],
                                obstacles[w]['p0'], obstacles[w]['p1'])
    if h < 0:
        t = rotate270(n)  # Tangent
        v = agents[i]['velocity']
        force = force_contact(h, n, v, t, agents[i]['mu'],
                              agents[i]['kappa'], agents[i]['damping'])

        agents[i]['force'][:] += force


@numba.jit(void(i8, i8, typeof(agent_type_three_circle)[:],
                typeof(obstacle_type_linear)[:]),
           nopython=True, nogil=True, cache=True)
def agent_obstacle_interaction_three_circle(i, w, agents, obstacles):
    """Interaction between three circle agent and line obstacle."""
    x_i = (agents[i]['position'], agents[i]['position_ls'], agents[i]['position_rs'])
    r_i = (agents[i]['r_t'], agents[i]['r_s'], agents[i]['r_s'])

    h, n, r_moment = distance_three_circle_line(x_i, r_i,
                                                obstacles[w]['p0'], obstacles[w]['p1'])
    if h < 0:
        t = rotate270(n)  # Tangent
        v = agents[i]['velocity']
        force = force_contact(h, n, v, t, agents[i]['mu'],
                              agents[i]['kappa'], agents[i]['damping'])

        agents[i]['force'][:] += force
        agents[i]['torque'] += cross(r_moment, force)


# Full interactions


@numba.jit(void(typeof(agent_type_circular)[:], i8[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_brute_circular(agents, indices):
    """Compute interaction forces between set of agents using brute force."""
    for l, i in enumerate(indices[:-1]):
        for j in indices[l + 1:]:
            agent_agent_interaction_circle(i, j, agents)


@numba.jit(void(typeof(agent_type_three_circle)[:], i8[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_brute_three_circle(agents, indices):
    """Compute interaction forces between set of agents using brute force."""
    for l, i in enumerate(indices[:-1]):
        for j in indices[l + 1:]:
            agent_agent_interaction_three_circle(i, j, agents)


@numba.jit(void(typeof(agent_type_circular)[:], i8[:], i8[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_brute_disjoint_circular(agents, indices_0, indices_1):
    """Compute interaction forces between two disjoint sets and of agents using
    brute force. Disjoint sets ``indices_0 ∩ indices_1 = ∅``."""
    for i in indices_0:
        for j in indices_1:
            agent_agent_interaction_circle(i, j, agents)


@numba.jit(void(typeof(agent_type_three_circle)[:], i8[:], i8[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_brute_disjoint_three_circle(agents, indices_0, indices_1):
    """Compute interaction forces between two disjoint sets and of agents using
    brute force. Disjoint sets ``indices_0 ∩ indices_1 = ∅``."""
    for i in indices_0:
        for j in indices_1:
            agent_agent_interaction_three_circle(i, j, agents)


@numba.jit(void(typeof(agent_type_circular)[:],
                i8[:], i8[:], i8[:], i8[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_block_list_circular(agents, index_list, count, offset, shape):
    """Iteration over all agents using block list algorithm."""
    # Neighbouring blocks
    nb = np.array(((1, 0), (1, 1), (0, 1), (1, -1)), dtype=np.int64)
    n, m = shape
    for i in range(n):
        for j in range(m):
            # Forces between agents inside the block
            ilist = get_block((i, j), index_list, count, offset, shape)
            agent_agent_brute_circular(agents, ilist)

            # Forces between agent inside the block and neighbouring agents
            for k in range(len(nb)):
                i2, j2 = nb[k]
                if 0 <= (i + i2) < n and 0 <= (j + j2) < m:
                    ilist2 = get_block((i + i2, j + j2), index_list, count,
                                       offset, shape)
                    agent_agent_brute_disjoint_circular(agents, ilist, ilist2)


@numba.jit(void(typeof(agent_type_three_circle)[:],
                i8[:], i8[:], i8[:], i8[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_block_list_three_circle(agents, index_list, count, offset, shape):
    """Iteration over all agents using block list algorithm."""
    # Neighbouring blocks
    nb = np.array(((1, 0), (1, 1), (0, 1), (1, -1)), dtype=np.int64)
    n, m = shape
    for i in range(n):
        for j in range(m):
            # Forces between agents inside the block
            ilist = get_block((i, j), index_list, count, offset, shape)
            agent_agent_brute_three_circle(agents, ilist)

            # Forces between agent inside the block and neighbouring agents
            for k in range(len(nb)):
                i2, j2 = nb[k]
                if 0 <= (i + i2) < n and 0 <= (j + j2) < m:
                    ilist2 = get_block((i + i2, j + j2), index_list, count,
                                       offset, shape)
                    agent_agent_brute_disjoint_three_circle(agents,
                                                            ilist,
                                                            ilist2)


def agent_agent_block_list(agents):
    index_list, count, offset, shape = block_list(agents['position'], SIGTH_SOC)
    if is_model(agents, 'circular'):
        agent_agent_block_list_circular(agents, index_list, count, offset, shape)
    elif is_model(agents, 'three_circle'):
        agent_agent_block_list_three_circle(agents, index_list, count, offset, shape)


def agent_agent_block_list_multithreaded(agent):
    pass


@numba.jit(void(typeof(agent_type_circular)[:], typeof(obstacle_type_linear)[:]),
           nopython=True, nogil=True, cache=True)
def circular_agent_linear_wall(agents, obstacles):
    """Agent wall"""
    for i in range(len(agents)):
        for w in range(len(obstacles)):
            agent_obstacle_interaction_circle(i, w, agents, obstacles)


@numba.jit(void(typeof(agent_type_three_circle)[:], typeof(obstacle_type_linear)[:]),
           nopython=True, nogil=True, cache=True)
def three_circle_agent_linear_wall(agents, obstacles):
    """Agent wall"""
    for i in range(len(agents)):
        for w in range(len(obstacles)):
            agent_obstacle_interaction_three_circle(i, w, agents, obstacles)


def agent_obstacle(agents, obstacles):
    if is_model(agents, 'circular'):
        circular_agent_linear_wall(agents, obstacles)
    elif is_model(agents, 'three_circle'):
        three_circle_agent_linear_wall(agents, obstacles)
