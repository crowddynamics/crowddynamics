r"""
Interactions
------------
Mathematically agent-agent and agent-obstacle interactions are governed by
summing over all agent-agent or agent-obstacle pairs. This is analogous to
n-body problems in physics.

Interactions between agents

.. math::
   \mathbf{f}_{i}^{agent-agent} =
   \sum_{j\neq i}^{} \left(\mathbf{f}_{ij}^{soc} + \mathbf{f}_{ij}^{c}\right)

.. math::
   M_{i}^{agent-agent} =
   \sum_{j\neq i}^{} \left(M_{ij}^{soc} + M_{ij}^{c}\right)


Interactions between agents and obstacles

.. math::
   \mathbf{f}_{i}^{agent-obstacles} = \sum_{w}^{} \mathbf{f}_{iw}^{c}

.. math::
   M_{i}^{agent-obstacles} = \sum_{w}^{} M_{iw}^{c}

"""
import numba
import numpy as np
from cell_lists import add_to_cells, neighboring_cells, iter_nearest_neighbors
from numba import void, i8, typeof
from numba.types import boolean

from crowddynamics.core.distance import distance_circles, \
    distance_circle_line, distance_three_circle_line, distance_three_circles
from crowddynamics.core.motion.contact import force_contact
from crowddynamics.core.motion.power_law import \
    force_social_circular, force_social_three_circle
from crowddynamics.core.structures import obstacle_type_linear
from crowddynamics.core.vector2D import rotate270, cross
from crowddynamics.exceptions import InvalidType
from crowddynamics.simulation.agents import agent_type_circular, \
    agent_type_three_circle, is_model

# TODO: load from config
SIGTH_SOC = 3.0


# Individual interactions


@numba.jit(void(i8, i8, typeof(agent_type_circular)[:]),
           nopython=True, nogil=True, cache=True)
def interaction_agent_agent_circular(i, j, agents):
    """Interaction between two circular agents."""
    h, n = distance_circles(agents[i]['position'], agents[i]['radius'],
                            agents[j]['position'], agents[j]['radius'])

    if h < SIGTH_SOC:
        force_i, force_j = force_social_circular(agents, i, j)

        if h < 0:
            t = rotate270(n)
            v = agents[i]['velocity'] - agents[j]['velocity']
            force_i += force_contact(h, n, v, t, agents[i]['mu'],
                                     agents[i]['kappa'], agents[i]['damping'])
            force_j -= force_contact(h, n, v, t, agents[j]['mu'],
                                     agents[j]['kappa'], agents[j]['damping'])

        agents[i]['force'][:] += force_i
        agents[j]['force'][:] += force_j


@numba.jit(void(i8, i8, typeof(agent_type_three_circle)[:]),
           nopython=True, nogil=True, cache=True)
def interaction_agent_agent_three_circle(i, j, agents):
    """Interaction between two three circle agents."""
    # Positions: center, left, right
    x_i = (agents[i]['position'], agents[i]['position_ls'],
           agents[i]['position_rs'])
    x_j = (agents[j]['position'], agents[j]['position_ls'],
           agents[j]['position_rs'])

    # Radii of torso and shoulders
    r_i = (agents[i]['r_t'], agents[i]['r_s'], agents[i]['r_s'])
    r_j = (agents[j]['r_t'], agents[j]['r_s'], agents[j]['r_s'])

    h, n, r_moment_i, r_moment_j = distance_three_circles(x_i, r_i, x_j, r_j)

    if h < SIGTH_SOC:
        force_i, force_j = force_social_three_circle(agents, i, j)

        if h < 0:
            t = rotate270(n)
            v = agents[i]['velocity'] - agents[j]['velocity']
            force_i += force_contact(h, n, v, t, agents[i]['mu'],
                                     agents[i]['kappa'], agents[i]['damping'])
            force_j -= force_contact(h, n, v, t, agents[j]['mu'],
                                     agents[j]['kappa'], agents[j]['damping'])

        agents[i]['force'][:] += force_i
        agents[j]['force'][:] += force_j

        agents[i]['torque'] += cross(r_moment_i, force_i)
        agents[j]['torque'] += cross(r_moment_j, force_j)


@numba.jit(void(i8, i8, typeof(agent_type_circular)[:],
                typeof(obstacle_type_linear)[:]),
           nopython=True, nogil=True, cache=True)
def interaction_agent_circular_obstacle(i, w, agents, obstacles):
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
def interaction_agent_three_circle_obstacle(i, w, agents, obstacles):
    """Interaction between three circle agent and line obstacle."""
    x_i = (agents[i]['position'], agents[i]['position_ls'],
           agents[i]['position_rs'])
    r_i = (agents[i]['r_t'], agents[i]['r_s'], agents[i]['r_s'])

    h, n, r_moment = distance_three_circle_line(
        x_i, r_i, obstacles[w]['p0'], obstacles[w]['p1'])
    if h < 0:
        t = rotate270(n)  # Tangent
        v = agents[i]['velocity']
        force = force_contact(h, n, v, t, agents[i]['mu'],
                              agents[i]['kappa'], agents[i]['damping'])

        agents[i]['force'][:] += force
        agents[i]['torque'] += cross(r_moment, force)


# Full interactions


@numba.jit(void(typeof(agent_type_circular)[:],
                i8[:], i8[:], i8[:], i8[:], i8[:], boolean[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_circular(agents, cell_indices, neigh_cells, points_indices,
                         cells_count, cells_offset, mask):
    for i, j in iter_nearest_neighbors(
            cell_indices, neigh_cells, points_indices, cells_count,
            cells_offset):
        if not mask[i] or not mask[j]:
            continue
        interaction_agent_agent_circular(i, j, agents)


@numba.jit(void(typeof(agent_type_three_circle)[:],
                i8[:], i8[:], i8[:], i8[:], i8[:], boolean[:]),
           nopython=True, nogil=True, cache=True)
def agent_agent_three_circle(agents, cell_indices, neigh_cells, points_indices,
                             cells_count, cells_offset, mask):
    for i, j in iter_nearest_neighbors(
            cell_indices, neigh_cells, points_indices, cells_count,
            cells_offset):
        if not mask[i] or not mask[j]:
            continue
        interaction_agent_agent_three_circle(i, j, agents)


@numba.jit(void(typeof(agent_type_circular)[:],
                typeof(obstacle_type_linear)[:], boolean[:]),
           nopython=True, nogil=True, cache=True)
def agent_circular_obstacle(agents, obstacles, mask):
    """Agent obstacle"""
    for i in range(len(agents)):
        if not mask[i]:
            continue
        for w in range(len(obstacles)):
            interaction_agent_circular_obstacle(i, w, agents, obstacles)


@numba.jit(void(typeof(agent_type_three_circle)[:],
                typeof(obstacle_type_linear)[:], boolean[:]),
           nopython=True, nogil=True, cache=True)
def agent_three_circle_obstacle(agents, obstacles, mask):
    """Agent obstacle"""
    for i in range(len(agents)):
        if not mask[i]:
            continue
        for w in range(len(obstacles)):
            interaction_agent_three_circle_obstacle(i, w, agents, obstacles)


# Higher level API

def agent_agent_block_list(agents, cell_size, mask):
    position = agents['position']
    points_indices, cells_count, cells_offset, grid_shape = add_to_cells(
        position, cell_size)

    cell_indices = np.arange(len(cells_count))
    neigh_cells = neighboring_cells(grid_shape)

    if is_model(agents, 'circular'):
        agent_agent_circular(agents, cell_indices, neigh_cells, points_indices,
                             cells_count, cells_offset, mask)
    elif is_model(agents, 'three_circle'):
        agent_agent_three_circle(agents, cell_indices, neigh_cells,
                                 points_indices, cells_count, cells_offset,
                                 mask)
    else:
        raise InvalidType


def agent_obstacle(agents, obstacles, mask):
    if is_model(agents, 'circular'):
        agent_circular_obstacle(agents, obstacles, mask)
    elif is_model(agents, 'three_circle'):
        agent_three_circle_obstacle(agents, obstacles, mask)
    else:
        raise InvalidType
