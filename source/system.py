from timeit import default_timer as timer

from source.core.integrator import euler_method
from source.parameters import goal_point
from source.struct.result import Result


class System:
    # TODO: AOT compilation
    # TODO: Adaptive Euler method
    # TODO: Optional walls

    def __init__(self, constant, agent, wall, dt=0.01):
        self.constant = constant
        self.agent = agent
        self.wall = wall
        self.result = Result()
        self.dt = dt
        self.integrator = euler_method(self.result, self.constant, self.agent,
                                       self.wall, self.dt)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.agent.set_goal_direction(goal_point)
            start = timer()
            ret = next(self.integrator)
            t_diff = timer() - start
            self.result.increment_wall_time(t_diff)
            if self.result.iterations % 100 == 0:
                print(self.result.iterations, ":", self.result.avg_wall_time())
            return ret
        except GeneratorExit:
            raise StopIteration()
