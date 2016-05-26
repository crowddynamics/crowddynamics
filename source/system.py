from timeit import default_timer as timer

from simulations.bottleneck.config import goal_point
from source.core.integrator import euler_method
from source.display import format_time
from source.struct.result import Result


class System:
    def __init__(self, constant, agent, wall, goal_area=None):
        # TODO: Multiple Optional walls
        # TODO: Multiple Goal areas
        self.constant = constant
        self.agent = agent
        self.wall = wall
        self.goal_area = goal_area

        self.result = Result(agent.size)

        # System
        self.integrator = euler_method(self.result, self.constant, self.agent,
                                       self.wall)

    def print_stats(self):
        out = "i: {:06d} | {:04d} | {} | {}".format(
            self.result.iterations,
            self.result.agents_in_goal,
            format_time(self.result.avg_wall_time()),
            format_time(self.result.wall_time_tot),
        )
        print(out)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # TODO: Goal direction updating
            self.agent.set_goal_direction(goal_point)

            # Execution timing
            start = timer()
            ret = next(self.integrator)
            t_diff = timer() - start
            self.result.increment_wall_time(t_diff)

            # Printing
            if self.result.iterations % 10 == 0:
                self.print_stats()

            # Check goal
            if self.goal_area is not None:
                num = self.goal_area.is_reached_by(self.agent)
                for _ in range(num):
                    if self.result.increment_agent_in_goal():
                        self.print_stats()
                        # raise StopIteration

            return ret
        except GeneratorExit:
            raise StopIteration()
