import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle

from source.display import format_time
from source.io.path import default_path

try:
    import seaborn
    seaborn.set()
except ImportError():
    pass


def plot_animation(simulation, x_dims, y_dims, save=False, frames=None):
    agent = simulation.agent
    constant = simulation.constant
    result = simulation.result

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set(xlim=x_dims, ylim=y_dims, xlabel=r'$ x $', ylabel=r'$ y $')

    # Text
    simu_time = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    goal_text = ax.text(0.02, 0.93, '', transform=ax.transAxes)
    wall_time = ax.text(0.02, 0.91, '', transform=ax.transAxes)

    # Walls
    wall = ax.add_collection(LineCollection(simulation.wall.linear_params))

    # Agents
    line, = ax.plot([], [], marker='o', lw=0, alpha=0.5)

    def _text():
        txt1 = "Simu Time = {:f}".format(result.t_tot)
        txt2 = "Goal = {:d} / {:d}".format(result.agents_in_goal, result.size)
        txt3 = "Real Time = " + format_time(result.wall_time_tot)

        simu_time.set_text(txt1)
        goal_text.set_text(txt2)
        wall_time.set_text(txt3)

    def _agent():
        # c = PatchCollection((Circle(x, r) for r, x in zip(agent.radius,
        #                                                   agent.position)))
        # return ax.add_collection(c)
        line.set_data(agent.position.T)

    def init():
        _text()
        _agent()
        return line, wall, simu_time, goal_text, wall_time

    def animate(i):
        next(simulation)
        _text()
        _agent()
        return line, wall, simu_time, goal_text, wall_time

    anim = FuncAnimation(fig, animate, init_func=init, interval=10, blit=True,
                         frames=frames, save_count=frames)

    if save:
        fps = round(1 / constant.dt)
        fname = default_path('animation.mp4', 'simulations', 'animations')
        anim.save(fname, fps=fps, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()
