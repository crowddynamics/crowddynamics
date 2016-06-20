from collections import Iterable

from .core.integrator import integrator
from .io.attributes import Intervals, Attrs, Attr
from .io.save import Save
from .struct.agent import agent_attr_names
from .struct.constant import constant_attr_names
from .struct.result import Result, result_attr_names
from .struct.wall import wall_attr_names


class Simulation:
    """
    Class for initialising and running a crowd simulation.
    """
    def __init__(self, constant, agent, wall=None, goals=None,
                 dirpath=None, name=None):
        # Make iterables and filter None values
        def _filter_none(arg):
            if not isinstance(arg, Iterable):
                arg = (arg,)
            return tuple(filter(None, arg))

        # Struct
        self.constant = constant
        self.agent = agent
        self.wall = _filter_none(wall)
        self.goals = _filter_none(goals)
        self.result = Result()

        # Integrator for rotational and spatial motion.
        self.integrator = self.result.computation_timer(integrator)

        # Interval for printing the values in result during the simulation.
        self.interval = Intervals(1.0)

        # Simulation IO for saving generated data to HDF5 file for analysis
        # and resuming a simulation.
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

    def advance(self):
        """
        :return: False is simulation ends otherwise True.
        """
        self.integrator(self.result, self.constant, self.agent, self.wall)

        # Check if agent are inside of bounds
        # TODO: Implementation

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
            self.exit()
            return False
        return True

    def exit(self):
        # Simulation exit
        print(self.result)
        self.save.hdf(self.result, self.attrs_result)
        for saver in self.savers:
            saver(brute=True)

    def run(self, iterations=None):
        """

        :param iterations: Execute simulation until number of iterations has been reached or if None run until simulation ends.
        :return: None
        """
        if iterations is None:
            while self.advance():
                pass
        else:
            while self.advance() and iterations > 0:
                iterations -= 1
