from collections import Iterable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Rectangle

from crowd_dynamics.area import GoalRectangle
from ..display import format_time
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

    # Figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set(xlim=x_dims, ylim=y_dims, xlabel=r'$ x $', ylabel=r'$ y $')
    ax.set_aspect("equal")

    # Text
    simu_time = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    goal_text = ax.text(0.02, 0.93, '', transform=ax.transAxes)
    wall_time = ax.text(0.02, 0.91, '', transform=ax.transAxes)

    # Agents
    agents = ax.scatter(agent.position[:, 0],
                        agent.position[:, 1],
                        s=400 * agent.radius**2,
                        alpha=0.8,
                        marker='o',
                        )

    # agents = [Circle(xy, rad) for xy, rad in zip(agent.position, agent.radius)]
    # agents = ax.add_collection(PatchCollection(agents))

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

    args += (ax.add_collection(PatchCollection(tuple(filter(None, areas)),
                                               alpha=0.2)), )

    def _text():
        txt1 = "Simulation: " + format_time(result.simulation_time)
        txt2 = "In goal: {:3d} / {:d}".format(result.in_goal, agent.size)
        txt3 = "Computation: " + format_time(result.computation_time_tot)

        simu_time.set_text(txt1)
        goal_text.set_text(txt2)
        wall_time.set_text(txt3)

    def _agent():
        agents.set_offsets(agent.position)

    def init():
        _text()
        _agent()
        return args

    def animate(i):
        if not simulation.advance() and not save:
            exit()
        _text()
        _agent()
        return args

    anim = FuncAnimation(fig, animate, init_func=init, interval=1, blit=True,
                         frames=frames, save_count=frames)

    if save:
        fps = round(1 / constant.dt)
        anim.save(filepath, fps=fps, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()
