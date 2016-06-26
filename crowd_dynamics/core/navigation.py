import numba

from .vector2d import normalize_nx2, angle_nx2


def navigation(agent, angle_update=None, direction_update=None):
    """
    Function for updating target angle and target direction.

    :param agent: Argument for functions
    :param angle_update: Function, or numpy array for updating agent.target_angle
    :param direction_update: Function, or numpy array for updating agent.target_direction
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


def exit_selection():
    """Exit selection policy."""
    pass


def direction_to_target_angle(agent):
    """:return: Angle of agent.target_direction."""
    return angle_nx2(agent.target_direction)


@numba.jit(nopython=True, nogil=True)
def set_goal_direction(agent, goal):
    """:return: Unit vector pointing from agents position to goal point."""
    return normalize_nx2(goal - agent.position)
