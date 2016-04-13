from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
from matplotlib import animation as animation


def func_plot(simulation, agents, field, frames=None, save=False):
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set(xlim=field['x_dims'],
           ylim=field['y_dims'],
           xlabel=r'$ x $',
           ylabel=r'$ y $')
    # TODO: cmap?
    # area = 20 * agents['radius']
    line, = ax.plot([], [], marker='o', lw=0, alpha=0.5)
    # line = ax.scatter([], [], s=area, alpha=0.5)

    def init_lines():
        positions = agents['position']
        line.set_data(positions.T)
        return line,

    def update_lines(i):
        agents = next(simulation)
        line.set_data(agents['position'].T)
        return line,

    anim = animation.FuncAnimation(fig, update_lines,
                                   init_func=init_lines,
                                   frames=frames,
                                   interval=1,
                                   blit=True)

    if save:
        writer = animation.writers['ffmpeg']
        writer = writer(fps=30, bitrate=1800)
        path = '/home/jaan/Dropbox/Projects/Crowd-Dynamics/animations/'
        anim.save(path + 'power_law.mp4', writer=writer)

    plt.show()
