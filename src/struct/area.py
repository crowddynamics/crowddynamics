from collections import OrderedDict

import numpy as np
import numba


goal_rec_spec = OrderedDict(
    center=numba.float64[:],
    radius=numba.float64[:],
)


@numba.jitclass(goal_rec_spec)
class GoalRectangle(object):
    def __init__(self, center, radius):
        # Vectors of shape=(2,)
        self.center = center  # (x, y)
        self.radius = radius  # (rx, ry)
        # self.angle = 0

    def is_reached_by(self, agent):
        vec = np.abs(self.center - agent.position) <= self.radius
        condition = vec[:, 0] & vec[:, 1]
        prev = np.sum(agent.goal_reached)
        agent.goal_reached |= condition
        num = np.sum(agent.goal_reached) - prev
        return num
