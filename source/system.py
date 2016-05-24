import math
import sys
from timeit import default_timer as timer

from simulations.one_exit_box.config import goal_point
from source.core.integrator import euler_method
from source.struct.result import Result


def _format_time(timespan, precision=3):
    """Jupyter notebook timeit time formatting.
    Formats the timespan in a human readable form"""

    if timespan >= 60.0:
        # we have more than a minute, format that in a human readable form
        # Idea from http://snipplr.com/view/5713/
        parts = [("d", 60 * 60 * 24), ("h", 60 * 60), ("min", 60), ("s", 1)]
        time = []
        leftover = timespan
        for suffix, length in parts:
            value = int(leftover / length)
            if value > 0:
                leftover = leftover % length
                time.append(u'%s%s' % (str(value), suffix))
            if leftover < 1:
                break
        return " ".join(time)

    # Unfortunately the unicode 'micro' symbol can cause problems in
    # certain terminals.
    # See bug: https://bugs.launchpad.net/ipython/+bug/348466
    # Try to prevent crashes by being more secure than it needs to
    # E.g. eclipse is able to print a µ, but has no sys.stdout.encoding set.
    units = [u"s", u"ms", u'us', "ns"]  # the save value
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
        try:
            u'\xb5'.encode(sys.stdout.encoding)
            units = [u"s", u"ms", u'\xb5s', "ns"]
        except:
            pass
    scaling = [1, 1e3, 1e6, 1e9]

    if timespan > 0.0:
        order = min(-int(math.floor(math.log10(timespan)) // 3), 3)
    else:
        order = 3
    return u"%.*g %s" % (precision, timespan * scaling[order], units[order])


class System:
    def __init__(self, constant, agent, wall, goal_area=None):
        # TODO: Multiple Optional walls
        self.constant = constant
        self.agent = agent
        self.wall = wall
        self.goal_area = goal_area

        # Results
        self.result = Result(agent.size)

        # System
        self.integrator = euler_method(self.result, self.constant, self.agent,
                                       self.wall)

    def print_stats(self):
        out = "i: {:06d} | {:04d} | {} | {}".format(
            self.result.iterations,
            self.result.agents_in_goal,
            _format_time(self.result.avg_wall_time()),
            _format_time(self.result.wall_time_tot),
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
                        raise StopIteration

            return ret
        except GeneratorExit:
            raise StopIteration()
