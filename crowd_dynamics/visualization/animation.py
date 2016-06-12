from collections import Iterable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Rectangle

from ..display import format_time
from ..struct.area import GoalRectangle
from ..struct.wall import LinearWall, RoundWall


try:
    import seaborn

    seaborn.set()
except ImportError():
    pass


def animation(simulation, x_dims, y_dims, save=False, frames=None,
              filepath=None):
    agent = simulation.agent
    constant = simulation.constant
    result = simulation.result

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

    if isinstance(simulation.wall, Iterable):
        walls = map(_wall, simulation.wall)
    else:
        walls = (_wall(simulation.wall),)

    args += tuple(filter(None, walls))

    def _area(area):
        if isinstance(area, GoalRectangle):
            return Rectangle(area.center - area.radius, 2 * area.radius[0],
                             2 * area.radius[1])
        else:
            return None

    if isinstance(simulation.goals, Iterable):
        areas = map(_area, simulation.goals)
    else:
        areas = (_area(simulation.goals),)

    args += (ax.add_collection(PatchCollection(tuple(filter(None, areas)), alpha=0.2)), )

    def _text():
        txt1 = "Simu Time = {:f}".format(result.simu_time_tot)
        txt2 = "Goal = {:d} / {:d}".format(result.agents_in_goal, result.size)
        txt3 = "Real Time = " + format_time(result.wall_time_tot)

        simu_time.set_text(txt1)
        goal_text.set_text(txt2)
        wall_time.set_text(txt3)

    def _agent():
        # c = PatchCollection((Circle(x, r) for r, x in zip(agent.radius,
        #                                                   agent.position)))
        # return ax.add_collection(c)
        agents.set_data(agent.position.T)

    def init():
        _text()
        _agent()
        return args

    def animate(i):
        next(simulation)
        _text()
        _agent()
        return args

    anim = FuncAnimation(fig, animate, init_func=init, interval=1, blit=True,
                         frames=frames, save_count=frames)

    if save:
        fps = round(1 / constant.dt)
        try:
            anim.save(filepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        except:
            pass
    else:
        plt.show()
