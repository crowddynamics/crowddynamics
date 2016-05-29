from collections import Iterable
from timeit import default_timer as timer

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Rectangle

from src.core.integrator import euler_method, euler_method2, euler_method0
from src.display import format_time
from src.struct.area import GoalRectangle
from src.struct.result import Result
from src.struct.wall import LinearWall, RoundWall


class System:
    def __init__(self, constant, agent, wall=None, goals=None):
        self.constant = constant
        self.agent = agent
        self.wall = wall
        self.goals = goals

        if not isinstance(self.wall, Iterable):
            self.wall = (self.wall,)
        self.wall = tuple(filter(None, self.wall))

        if not isinstance(self.goals, Iterable):
            self.goals = (self.goals,)
        self.goals = tuple(filter(None, self.goals))

        self.result = Result(agent.size)

        # System
        if len(self.wall) == 0:
            self.integrator = euler_method0(self.result,
                                            self.constant,
                                            self.agent)
        if len(self.wall) == 1:
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

        self.prev_time = 0

    def animation(self, x_dims, y_dims, save=False, frames=None, filepath=None):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set(xlim=x_dims, ylim=y_dims, xlabel=r'$ x $', ylabel=r'$ y $')
        ax.set_aspect("equal")

        # Text
        simu_time = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        goal_text = ax.text(0.02, 0.93, '', transform=ax.transAxes)
        wall_time = ax.text(0.02, 0.91, '', transform=ax.transAxes)

        # Agents
        agents, = ax.plot([], [], marker='o', lw=0, alpha=0.5)

        args = (simu_time, goal_text, wall_time, agents)

        # Walls
        def _wall(wall):
            if isinstance(wall, LinearWall):
                return ax.add_collection(LineCollection(wall.params))
            elif isinstance(wall, RoundWall):
                c = (Circle(*wall.deconstruct(i)) for i in range(wall.size))
                return ax.add_collection(PatchCollection(c))
            else:
                return None

        if isinstance(self.wall, Iterable):
            walls = map(_wall, self.wall)
        else:
            walls = (_wall(self.wall),)

        args += tuple(filter(None, walls))

        def _area(area):
            if isinstance(area, GoalRectangle):
                return Rectangle(area.center - area.radius, 2 * area.radius[0],
                                 2 * area.radius[1])
            else:
                return None

        if isinstance(self.goals, Iterable):
            areas = map(_area, self.goals)
        else:
            areas = (_area(self.goals),)

        args += (ax.add_collection(
            PatchCollection(tuple(filter(None, areas)), alpha=0.2)),)

        def _text():
            txt1 = "Simu Time = {:f}".format(self.result.t_tot)
            txt2 = "Goal = {:d} / {:d}".format(self.result.agents_in_goal,
                                               self.result.size)
            txt3 = "Real Time = " + format_time(self.result.wall_time_tot)

            simu_time.set_text(txt1)
            goal_text.set_text(txt2)
            wall_time.set_text(txt3)

        def _agent():
            # c = PatchCollection((Circle(x, r) for r, x in zip(agent.radius,
            #                                                   agent.position)))
            # return ax.add_collection(c)
            agents.set_data(self.agent.position.T)

        def init():
            _text()
            _agent()
            return args

        def animate(i):
            self.__next__()
            _text()
            _agent()
            return args

        anim = FuncAnimation(fig, animate, init_func=init, interval=10,
                             blit=True,
                             frames=frames, save_count=frames)
        if save:
            fps = round(1 / self.constant.dt)
            anim.save(filepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        else:
            plt.show()

    def exhaust(self):
        for _ in self:
            pass

    def save_results(self):
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
            if self.result.wall_time_tot - self.prev_time > 1.0:
                self.prev_time = self.result.wall_time_tot
                self.print_stats()

            # Check goal
            for goal in self.goals:
                self.goal_reached(goal)

            return ret
        except GeneratorExit:
            raise StopIteration()
