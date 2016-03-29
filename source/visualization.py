from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import seaborn as sns
from matplotlib import animation as animation

from source.simulation import init_simulation, rad, size, positions, simu


def visualization(frames):
    init_simulation()

    writer = animation.writers['ffmpeg']
    writer = writer(fps=30, bitrate=1800)
    path = '/home/jaan/Dropbox/Projects/Crowd-Dynamics/animations/'

    ms = rad * 300 / size

    fig, ax = plt.subplots(ncols=2, figsize=(16, 10))
    sns.set()

    ax[0].set(xlim=(0, size), ylim=(0, size),
              xlabel=r'$ x $', ylabel=r'$ y $')

    ax[1].set(xlim=(0, frames), ylim=(0, 10),
              xlabel=r'$ t $', ylabel=r'$ F $')

    line, = ax[0].plot([], [], lw=0, markersize=ms, marker='o', alpha=0.5)
    line2, = ax[1].plot([], [], lw=0, markersize=2, marker='o', alpha=0.5)

    ones = np.ones_like(positions)

    def init():
        line.set_data(positions.T)
        line2.set_data(ones, np.zeros_like(ones))
        return line, line2

    def update_line(i):
        x, f = next(simu)
        f = np.abs(f)
        line.set_data(x.T)
        line2.set_data(i * ones, f)
        return line, line2

    anim = animation.FuncAnimation(fig, update_line,
                                   init_func=init,
                                   frames=frames,
                                   interval=1,  #500/30,
                                   blit=True)

    # anim.save(path + 'power_law.mp4', writer=writer)
    plt.show()