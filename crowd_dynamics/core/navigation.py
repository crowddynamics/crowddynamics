import numpy as np
import numba

from .vector2d import normalize_vec


"""
Function that returns unit vector for agent.goal_direction

Vector field
- Incompressible, irrotational and inviscosid fluid flow
- Poisson equation, Heat equation, Navier-Stokes
"""


def exit_selection():
    """Exit selection policy."""
    pass


def set_goal_direction(agent, goal):
    """Update goal direction for agent that have not reached their goals."""
    mask = agent.goal_reached ^ True
    if np.sum(mask):
        agent.goal_direction[mask] = normalize_vec(goal - agent.position[mask])


@numba.jit(nopython=True, nogil=True)
def direction_to_target_angle(agent):
    agent.target_angle = np.arctan2(agent.target_direction[:, 0],
                                    agent.target_direction[:, 1])


def navigation(agent, goal_point):
    # TODO: Navigation
    set_goal_direction(agent, goal_point)

