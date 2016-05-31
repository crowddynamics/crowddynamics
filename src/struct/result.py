import numpy as np
from collections import OrderedDict

from numba import float64, jitclass, int64


spec_result = OrderedDict(
    iterations=int64,
    t_init=float64,
    t_tot=float64,
    wall_time_tot=float64,
    agents_in_goal=int64,
    agents_in_goal_times=float64[:],
    size=int64,
)


@jitclass(spec_result)
class Result(object):
    """
    Struct for simulation results.
    """

    def __init__(self, size):
        # Properties
        self.size = size
        # Wall time (Time spent computing)
        self.wall_time_init = 0
        self.wall_time_tot = 0
        # Simulation data
        self.iterations = 0
        self.simu_time_tot = 0
        self.agents_in_goal = 0
        self.agents_in_goal_times = np.zeros(self.size)

    def increment_simu_time(self, dt):
        self.simu_time_tot += dt
        self.iterations += 1

    def increment_wall_time(self, t_diff):
        if self.iterations == 0:
            # Initial time difference is higher because jit compilation
            self.wall_time_init = t_diff
        else:
            self.wall_time_tot += t_diff

    def increment_agent_in_goal(self):
        self.agents_in_goal_times[self.agents_in_goal] = self.simu_time_tot
        self.agents_in_goal += 1
        if self.agents_in_goal != self.size:
            return 0
        else:
            return 1

    def avg_wall_time(self):
        if self.iterations > 0:
            return self.wall_time_tot / (self.iterations - 1)
        else:
            return 0
