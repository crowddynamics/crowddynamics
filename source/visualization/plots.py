import matplotlib.pyplot as plt
from matplotlib import animation as animation
from matplotlib.collections import LineCollection

from source.io.path import default_path

try:
    import seaborn

    seaborn.set()
except ImportError():
    pass


def plot_animation(simulation, x_dims, y_dims, save=False):
    """
    http://matplotlib.org/1.4.1/examples/animation/index.html
    http://matplotlib.org/examples/api/patch_collection.html
    """

    agent = simulation.agent
    linear_wall = simulation.wall
    constant = simulation.constant

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set(xlim=x_dims, ylim=y_dims, xlabel=r'$ x $', ylabel=r'$ y $')
    line, = ax.plot([], [], marker='o', lw=0, alpha=0.5)

    line_segments = LineCollection(linear_wall.linear_params)
    coll = ax.add_collection(line_segments)

    def init():
        line.set_data(agent.position.T)
        return line, coll

    def animate(i):
        next(simulation)
        line.set_data(agent.position.T)
        return line, coll

    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   interval=10,
                                   blit=True)

    if save:
        fps = round(1 / constant.dt)
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        fname = default_path('animation.mp4', 'simulations', 'animations')
        anim.save(fname, writer=writer, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()
