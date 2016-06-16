import numpy as np


class GoalRectangle(object):
    def __init__(self, center, radius):
        self.center = center  # (x, y)
        self.radius = radius  # (rx, ry)

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
