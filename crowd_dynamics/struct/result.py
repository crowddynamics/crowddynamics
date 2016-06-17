from collections import deque
from functools import wraps
from timeit import default_timer as timer

import numpy as np

from ..display import format_time

result_attr_names = (
    "computation_time_high",
    "computation_time_tot",
    "iterations",
    "simulation_time",
    "time_steps",
    "in_goal_time",
)


class Result(object):
    """
    Struct for simulation results.
    """

    def __init__(self, deque_maxlen=100):
        self.initial_flag = True

        # Simulation data
        self.iterations = 0
        self.simulation_time = 0
        self.time_steps = []
        self.in_goal = 0
        self.in_goal_time = []

        # Real time
        self.computation_time_high = 0
        self.computation_time_tot = 0
        self.computation_time = deque(maxlen=deque_maxlen)

    def increment_in_goal_time(self):
        self.in_goal += 1
        self.in_goal_time.append(self.simulation_time)

    def increment_simulation_time(self, dt):
        self.iterations += 1
        self.simulation_time += dt
        self.time_steps.append(dt)

    def increment_computation_time(self, dt, tol=1.0):
        """Does not record times higher than tol."""
        if dt < tol:
            self.computation_time_tot += dt
            self.computation_time.append(dt)
        else:
            self.computation_time_high += dt

    def computation_timer(self, func, tol=1.0):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = timer()
            ret = func(*args, **kwargs)
            dt = timer() - start
            self.increment_computation_time(dt, tol)
            return ret
        return wrapper

    def __str__(self):
        if self.computation_time:
            mean_time = np.mean(self.computation_time)
        else:
            mean_time = 0

        if self.time_steps:
            mean_dt = np.mean(self.time_steps[:100])
        else:
            mean_dt = 0

        out = "i: {:6d} | {:4d} | {} | {} | {:4f}".format(
            self.iterations, self.in_goal, format_time(mean_time),
            format_time(self.computation_time_tot),
            mean_dt
        )
        if self.initial_flag:
            out = "Initial time: " + format_time(self.computation_time_high) + \
                  "\n" + out
            self.initial_flag = False
        return out
