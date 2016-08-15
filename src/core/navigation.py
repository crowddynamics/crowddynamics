import numpy as np
import skfmm

from .vector2d import angle_nx2


class Navigation:
    """Target direction"""
    def __init__(self, agent, domain, obstacles, exits):
        self.agent = agent
        self.domain = domain
        self.obstacles = obstacles
        self.exits = exits

    def distance_map(self, n=201, dx=1e-2):
        # Discretize the domain
        x = np.linspace(self.domain.x[0], self.domain.x[1], n)
        y = np.linspace(self.domain.y[0], self.domain.y[0], n)
        X, Y = np.meshgrid(x, y)

        # Set contour.
        phi = -1 * np.ones_like(X)

        # Exits. Zero contour defines the exit.
        # phi[] = 1

        # Obstacles are defined by masked values.
        mask = None

        phi = np.ma.MaskedArray(phi, mask)
        d_map = skfmm.distance(phi, dx=dx)

        return

    def static_potential(self):
        return

    def dynamic_potential(self):
        return


class Orientation:
    """Target orientation"""
    def __init__(self):
        pass


class ExitSelection:
    """Exit selection policy."""
    def __init__(self):
        pass


def direction_to_target_angle(agent):
    """:return: Angle of agent.target_direction."""
    return angle_nx2(agent.target_direction)


def navigator(agent,
              angle_update=direction_to_target_angle,
              direction_update=None):
    """
    Function for updating target angle and target direction.

    :param agent: Argument for functions
    :param angle_update: Function, or numpy array for updating agent.target_angle
    :param direction_update: Function, or numpy array for updating agent.target_direction
    """
    if direction_update is not None:
        if callable(direction_update):
            agent.target_direction[:] = direction_update(agent)
        else:
            agent.target_direction[:] = direction_update

    if angle_update is not None and agent.orientable:
        if callable(angle_update):
            agent.target_angle[:] = angle_update(agent)
        else:
            agent.target_angle[:] = angle_update

