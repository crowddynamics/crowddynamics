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
agents_num = 200
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
    mass = np.ones(agents_num)
    radius = rad * np.ones(agents_num)
    # Generator for new positions
    simu = update_positions(positions, velocities, goal_velocity, radius, mass)


def visualization():
    init_simulation()

    writer = animation.writers['ffmpeg']
    writer = writer(fps=30, bitrate=1800)
    path = '/home/jaan/Dropbox/Projects/Crowd-Dynamics/animations/'

    ms = rad * 300 / size

    fig, ax = plt.subplots()
    sns.set()
    ax.set(xlim=(0, size), ylim=(0, size), xlabel=r'$ x $', ylabel=r'$ y $')
    line, = ax.plot([], [], lw=0, markersize=ms, marker='o', alpha=0.5)

    def init():
        line.set_data(positions.T)
        return line,

    def update_line(i):
        x = next(simu)
        line.set_data(x.T)
        return line,

    anim = animation.FuncAnimation(fig, update_line,
                                   init_func=init,
                                   frames=1000,
                                   interval=1,  #500/30,
                                   blit=True)

    anim.save(path + 'power_law.mp4', writer=writer)
    # plt.show()


def profile(iterations):
    init_simulation()
    for _ in range(iterations):
        next(simu)


visualization()
# profile(100)
