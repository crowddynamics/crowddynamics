from collections import OrderedDict

from numba import float64, jitclass, int64


spec_result = OrderedDict(
    iterations=int64,
    t_tot=float64,
    wall_time_tot=float64,
)


@jitclass(spec_result)
class Result(object):
    """
    Struct for simulation results.
    """

    def __init__(self):
        self.iterations = 0
        self.t_tot = 0
        self.wall_time_tot = 0

    def increment(self, dt):
        self.t_tot += dt
        self.iterations += 1

    def increment_wall_time(self, t_diff):
        if self.iterations > 0:
            self.wall_time_tot += t_diff

    def avg_wall_time(self):
        if self.iterations > 0:
            return self.wall_time_tot / (self.iterations - 1)
        else:
            return 0
