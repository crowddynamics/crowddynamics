# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from source.calculator import update_positions


def init_simulation(agents_num, size):
    """

    :param agents_num:
    :param size:
    :return:
    """
    tau = 0.5
    # Arrays shape of (agent_num, 2) for x and y-components
    shape = (agents_num, 2)  # rows, cols
    angle = np.random.uniform(0, 2 * np.pi, agents_num)

    position = np.random.uniform(0, size, shape)
    velocity = np.stack((np.cos(angle), np.sin(angle)), axis=1)
    goal_velocity = 1.5 * np.copy(velocity)
    mass = np.ones(agents_num)
    rad = 0.2
    radius = rad * np.ones(agents_num)

    simu = update_positions(position, velocity, goal_velocity, radius, mass, tau)

    fig, ax = plt.subplots()
    ax.set(xlim=(0, size), ylim=(0, size), xlabel=r'$ x $', ylabel=r'$ y $')
    line, = ax.plot([], [], lw=0, markersize=rad * 300 / size,
                    marker='o', alpha=0.5)

    def init():
        line.set_data(position.T)
        return line,

    def update_plot(i):
        vals = next(simu)
        line.set_data(vals.T)
        return line,

    anim = animation.FuncAnimation(fig, update_plot, init_func=init,
                                   frames=500, interval=500/30, blit=True)
    plt.show()


init_simulation(40, 10)
