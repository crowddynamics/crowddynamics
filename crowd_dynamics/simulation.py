import operator
from collections import Iterable

from .core.integrator import integrator
from .io.attributes import Intervals, Attrs, Attr
from .io.save import Save
from .struct.agent import agent_attr_names
from .struct.constant import constant_attr_names
from .struct.result import Result, result_attr_names
from .struct.wall import wall_attr_names
from .visualization.animation import animation


class Simulation:
    def __init__(self, constant, agent, wall=None, goals=None, dirpath=None,
                 name=None, x=None, y=None):
        width = x[1] - x[0]
        height = y[1] - y[0]
        self.x_dims = x
        self.y_dims = y

        offset = abs(width - height) / 2
        if width > height:
            self.y_dims = y[0] - offset, y[1] + offset
        elif width < height:
            self.x_dims = x[0] - offset, x[1] + offset
        self.x_dims = tuple(map(operator.add, self.x_dims, (-5, 5)))
        self.y_dims = tuple(map(operator.add, self.y_dims, (-5, 5)))

        # Make iterables and filter None values
        def _filter_none(arg):
            if not isinstance(arg, Iterable):
                arg = (arg,)
            return tuple(filter(None, arg))

        # List of thing to implement
        # TODO: Better Visualization and movie writing
        # TODO: Elliptical/Three circle Agent model and orientation
        # TODO: Pathfinding/Exit selection algorithm
        # TODO: Game theoretical exit congestion algorithm
        # TODO: Result Analysis
        # TODO: New simulations
        # TODO: np.dot -> check performance and gil
        # TODO: check continuity -> numpy.ascontiguousarray
        # TODO: Tables of anthropometric data
        # TODO: Egress flow to goal areas

        # Struct
        self.constant = constant
        self.agent = agent
        self.wall = _filter_none(wall)
        self.goals = _filter_none(goals)
        self.result = Result()

        # Integrator for updating multi-agent system
        self.integrator = self.result.computation_timer(integrator)

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

        self.savers = _filter_none(
            [self.save.hdf(self.constant, self.attrs_constant),
             self.save.hdf(self.agent, self.attrs_agent)] +
            [self.save.hdf(w, self.attrs_wall) for w in self.wall]
        )

    def animation(self, fname=None, save=False, frames=None):
        if save:
            filepath = self.save.animation(fname)
        else:
            filepath = None
        animation(self, self.x_dims, self.y_dims, save, frames, filepath)

    def advance(self):
        """
        :return: False is simulation ends otherwise True.
        """
        self.integrator(self.result, self.constant, self.agent, self.wall)

        # Goals
        for goal in self.goals:
            num = goal.is_reached_by(self.agent)
            for _ in range(num):
                self.result.increment_in_goal_time()

        # Save
        for saver in self.savers:
            saver()

        # Display
        if self.interval():
            print(self.result)

        if len(self.result.in_goal_time) == self.agent.size:
            # Simulation exit
            print(self.result)
            self.save.hdf(self.result, self.attrs_result)
            for saver in self.savers:
                saver(brute=True)
            return False

        return True

    def run(self, iterations=None):
        """

        :param iterations: Run simulation until number of iterations is reached
        or if None run until simulation ends.
        :return:
        """
        if iterations is None:
            while self.advance():
                pass
        else:
            while self.advance() and iterations > 0:
                iterations -= 1
