from collections import Iterable
from timeit import default_timer as timer

from src.core.integrator import euler_method, euler_method2, euler_method0
from src.display import format_time
from src.io.attributes import Intervals, Attrs, Attr
from src.io.save import Save
from src.struct.agent import agent_attr_names
from src.struct.constant import constant_attr_names
from src.struct.result import Result, result_attr_names
from src.struct.wall import wall_attr_names
from src.visualization.animation import animation


class System:
    def __init__(self, constant, agent, wall=None, goals=None, dirpath=None,
                 name=None):
        # Make iterables and filter None values
        def _filter(arg):
            if not isinstance(arg, Iterable):
                arg = (arg,)
            return tuple(filter(None, arg))

        # List of thing to implement
        # TODO: Saving to hdf5
        # TODO: Social force wall
        # TODO: Better Visualization and movie writing
        # TODO: Elliptical/Three circle Agent model and orientation
        # TODO: Pathfinding/Exit selection algorithm
        # TODO: Game theoretical exit congestion algorithm
        # TODO: Result Analysis
        # TODO: New simulations

        # Struct
        self.constant = constant
        self.agent = agent
        self.wall = _filter(wall)
        self.goals = _filter(goals)
        self.result = Result(agent.size)

        # TODO: Limit iterations
        # Integrator for updating multi-agent system
        method = (euler_method0, euler_method, euler_method2)[len(self.wall)]
        self.integrator = method(self.result, self.constant, self.agent,
                                 *self.wall)

        # Object for saving simulation data
        self.interval = Intervals(1.0)
        self.save = Save(dirpath, name)
        self.attrs_constant = Attrs(constant_attr_names)
        self.attrs_result = Attrs(result_attr_names)
        self.attrs_agent = Attrs(agent_attr_names, Intervals(1.0))
        self.attrs_wall = Attrs(wall_attr_names)
        self.attrs_agent["position"] = Attr("position", True, True)
        self.attrs_agent["velocity"] = Attr("velocity", True, True)
        self.attrs_agent["force"] = Attr("force", True, True)

        self.hdf = _filter(
            [self.save.new_hdf_saver(self.constant, self.attrs_constant),
             self.save.new_hdf_saver(self.agent, self.attrs_agent)] +
            [self.save.new_hdf_saver(w, self.attrs_wall) for w in self.wall]
        )

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
            self.save.new_hdf_saver(self.result, self.attrs_result)
            raise StopIteration()

    def __iter__(self):
        return self
