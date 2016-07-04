import numpy as np


# TODO: Spring, Void


class Area(object):
    def __init__(self):
        pass

    def is_reached_by(self):
        pass


class Bounds:
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=np.float64)
        self.radius = np.array(radius, dtype=np.float64)

    def update(self, agent):
        """Agents outside bounds are updated to inactive state."""
        vec = np.abs(self.center - agent.position) <= self.radius
        condition = vec[:, 0] & vec[:, 1]
        agent.active &= condition


class Goal(object):
    def __init__(self, center, radius):
        """Rectangle shaped."""
        self.center = np.array(center, dtype=np.float64)
        self.radius = np.array(radius, dtype=np.float64)

    def update(self, agent):
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


