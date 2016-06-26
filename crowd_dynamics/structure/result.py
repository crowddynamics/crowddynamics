import numpy as np

result_attr_names = (
    "iterations",
    "simulation_time",
    "time_steps",
    "in_goal_time",
)


class Result(object):
    """
    Struct for simulation results.
    """

    def __init__(self):
        self.initial_flag = True

        # Simulation data
        self.iterations = 0
        self.simulation_time = 0
        self.time_steps = [0]
        self.in_goal = 0
        self.in_goal_time = []

    def increment_in_goal_time(self):
        self.in_goal += 1
        self.in_goal_time.append(self.simulation_time)

    def increment_simulation_time(self, dt):
        self.iterations += 1
        self.simulation_time += dt
        self.time_steps.append(dt)

    def __str__(self):
        # header -> result_attr_names
        values = (self.iterations, self.simulation_time, self.in_goal,
                  np.mean(self.time_steps[-100:]))
        out = "Iterations: {:6d} | Simu time: {:4f} | " \
              "In goal: {:4d} | Mean dt: {:4f}".format(*values)
        return out
