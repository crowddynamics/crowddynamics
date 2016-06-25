import numpy as np
import numba

from .vector2d import normalize_nx2, angle_nx2


def navigation(agent, angle_update=None, direction_update=None):
    """
    Function for updating target angle and target direction.
    :param agent:
    :param angle_update:
    :param direction_update:
    """
    if angle_update is not None and agent.orientable_flag:
        if callable(angle_update):
            agent.target_angle = angle_update(agent)
        else:
            agent.target_angle = angle_update

    if direction_update is not None:
        if callable(direction_update):
            agent.target_direction = direction_update(agent)
        else:
            agent.target_direction = direction_update


# @numba.jit(nopython=True, nogil=True)
def exit_selection():
    """Exit selection policy."""
    pass


@numba.jit(nopython=True, nogil=True)
def direction_to_target_angle(agent):
    return angle_nx2(agent.target_direction)


@numba.jit(nopython=True, nogil=True)
def set_goal_direction(agent, goal):
    """Update goal direction for agent that have not reached their goals."""
    mask = agent.goal_reached ^ True
    if np.sum(mask):
        agent.goal_direction[mask] = normalize_nx2(goal - agent.position[mask])

