from collections import Iterable
from timeit import default_timer as timer

from src.core.integrator import euler_method, euler_method2, euler_method0
from src.display import format_time
from src.io.save import Save
from src.struct.constant import constant_attr_names
from src.struct.result import Result, result_attr_names
from src.visualization.animation import animation


class Attr:
    def __init__(self, name, is_resizable=False, save_func=None):
        self.name = name
        self.is_resizable = is_resizable
        self.save_func = save_func

    def __str__(self):
        return self.name


class Intervals:
    def __init__(self, interval):
        self.interval = interval
        if self.interval < 0:
            raise ValueError("Interval should be > 0")
        self.prev = timer()

    def __call__(self, *args, **kwargs):
        if self.interval == 0:
            return True
        else:
            current = timer()
            diff = current - self.prev
            ret = diff >= self.interval
            if ret:
                self.prev = current
            return ret


class System:
    def __init__(self, constant, agent, wall=None, goals=None, dirpath=None,
                 name=None):
        # Struct
        self.constant = constant
        self.agent = agent
        self.wall = wall
        self.goals = goals

        # Make iterables from wall and goal and filter None values
        if not isinstance(self.wall, Iterable):
            self.wall = (self.wall,)
        self.wall = tuple(filter(None, self.wall))

        if not isinstance(self.goals, Iterable):
            self.goals = (self.goals,)
        self.goals = tuple(filter(None, self.goals))

        # Result struct
        self.result = Result(agent.size)

        # Integrator for updating multi-agent system
        self.integrator = None
        if len(self.wall) == 0:
            self.integrator = euler_method0(self.result,
                                            self.constant,
                                            self.agent)
        elif len(self.wall) == 1:
            self.integrator = euler_method(self.result,
                                           self.constant,
                                           self.agent,
                                           *self.wall)
        elif len(self.wall) == 2:
            self.integrator = euler_method2(self.result,
                                            self.constant,
                                            self.agent,
                                            *self.wall)
        else:
            raise ValueError()

        # Object for saving simulation data
        self.interval = Intervals(1.0)
        self.save = Save(dirpath, name)
        constant_attrs = [Attr(name) for name in constant_attr_names]
        agent_attrs = [
            Attr("mass", False, None),
            Attr("radius", False, None),
            Attr("position", True, Intervals(5)),
        ]
        self.hdf = tuple(filter(None, [
            self.save.to_hdf(self.constant, constant_attrs),
            self.save.to_hdf(self.agent, agent_attrs),
            # self.save.to_hdf(self.result),
        ]))

    def animation(self, x_dims, y_dims, fname=None, save=False, frames=None):
        if save:
            filepath = self.save.animation(fname)
        else:
            filepath = None
        animation(self, x_dims, y_dims, save, frames, filepath)

    def exhaust(self):
        for _ in self:
            pass

    def print_stats(self):
        out = "i: {:06d} | {:04d} | {} | {}".format(
            self.result.iterations,
            self.result.agents_in_goal,
            format_time(self.result.avg_wall_time()),
            format_time(self.result.wall_time_tot),
        )
        print(out)

    def goal_reached(self, goal):
        num = goal.is_reached_by(self.agent)
        for _ in range(num):
            if self.result.increment_agent_in_goal():
                self.print_stats()
                raise GeneratorExit()

    def __iter__(self):
        return self

    def __next__(self):
        """
        Generator exits when all agents have reached their goals.
        """
        try:
            # TODO: Goal direction updating
            # self.agent.set_goal_direction(goal_point)

            # Execution timing
            start = timer()
            ret = next(self.integrator)
            t_diff = timer() - start
            self.result.increment_wall_time(t_diff)

            # Printing
            if self.interval():
                self.print_stats()

            for s in self.hdf:
                next(s)

            # Check goal
            for goal in self.goals:
                self.goal_reached(goal)

            return ret
        except GeneratorExit:
            result_attrs = [Attr(name) for name in result_attr_names]
            self.save.to_hdf(self.result, result_attrs)
            raise StopIteration()
