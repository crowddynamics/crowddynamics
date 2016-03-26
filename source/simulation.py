# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Find "Source" module to perform import
module_path = '/home/jaan/Dropbox/Projects/Crowd-Dynamics/'
sys.path.append(module_path)

from source.core import update_positions


rad = 0.2
agents_num = 50
size = 8
positions = None
velocities = None
# TODO: Track distances


def init_simulation():
    """

    :param agents_num:
    :param size:
    :return:
    """
    # Arrays shape of (agent_num, 2) for x and y-components
    global positions, velocities, simu

    shape = (agents_num, 2)  # rows, cols
    angle = np.random.uniform(0, 2 * np.pi, agents_num)

    positions = np.random.uniform(0, size, shape)
    velocities = np.stack((np.cos(angle), np.sin(angle)), axis=1)
    goal_velocity = 1.5 * np.copy(velocities)
    masses = np.ones(agents_num)
    # masses = np.random.uniform(0.9, 1, agents_num)
    radii = rad * np.ones(agents_num)
    walls = None

    # Generator for new positions
    simu = update_positions(positions, velocities, goal_velocity, radii, masses)


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


def profile(iterations):
    init_simulation()
    for _ in range(iterations):
        next(simu)


visualization(1000)
# profile(100)
