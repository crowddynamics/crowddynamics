import numpy as np

# TODO: Spring, Void
from crowd_dynamics.core.vector2d import length


class Area(object):
    def __init__(self):
        pass

    def is_reached_by(self):
        pass


class Bounds:
    def __init__(self):
        pass


class Goal(object):
    def __init__(self, center, radius):
        """Rectangle shaped."""
        self.center = np.array(center, dtype=np.float64)
        self.radius = np.array(radius, dtype=np.float64)

    def is_reached_by(self, agent):
        """Updates agent that have reached goal.

        :param agent:
        :return: Number of agent that reached the goal.
        """
        vec = np.abs(self.center - agent.position) <= self.radius
        condition = vec[:, 0] & vec[:, 1]
        prev_num = np.sum(agent.goal_reached)
        agent.goal_reached |= condition
        num = np.sum(agent.goal_reached) - prev_num
        return num


class ExitDoor(object):
    def __init__(self, p0, p1, agent_radius):
        """Exit door / Bottleneck"""
        self.p = np.array((p0, p1))
        self.mid = (self.p[0] + self.p[1]) / 2
        self.radius = length(self.p[1] - self.p[0]) / 2
        self.capacity = 1.0 * self.radius // agent_radius
