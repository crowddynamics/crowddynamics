import os
from collections import Iterable, deque
from itertools import repeat, islice

import numpy as np

import matplotlib.lines as lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn
from matplotlib import animation as animation


def consume(iterator, n=None):
    """Advance the iterator n-steps ahead. If n is none, consume entirely."""
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def agent_patches(x, r, **kwargs):
    if not isinstance(r, Iterable):
        # If all agents have same radius
        r = repeat(r)
    patch = lambda *args: mpatches.Circle(*args, **kwargs)
    patches = map(patch, x, r)
    return patches


def linear_wall_patches(p, **kwargs):
    f = lambda arg: tuple(zip(*arg))
    data = tuple(map(f, p))
    patch = lambda args: lines.Line2D(*args, **kwargs)
    patches = map(patch, data)
    return patches


def force_patches(x, f, tol=0.0001, **kwargs):
    mag = np.hypot(f[:, 0], f[:, 1])
    mask = mag > tol
    x = x[mask]
    f = f[mask]
    patch = lambda *args: mpatches.Arrow(*args, **kwargs)
    patches = map(patch, x[:, 0], x[:, 1], f[:, 0], f[:, 1])
    return patches


def add_patches(ax, patches):
    consume(map(ax.add_artist, patches))


def plot_field(position, radius, x_dims, y_dims,
               linear_wall=None, force=None, save=True):
    seaborn.set()
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set(xlim=x_dims, ylim=y_dims, xlabel=r'$ x $', ylabel=r'$ y $')
    add_patches(ax, agent_patches(position, radius, alpha=0.5))

    if linear_wall is not None:
        add_patches(ax, linear_wall_patches(linear_wall, color='black'))

    if force is not None:
        add_patches(ax, force_patches(position, force, alpha=0.8, width=0.15))

    # Save figure to figures/field.pdf
    if save:
        root = '/home/jaan/Dropbox/Projects/Crowd-Dynamics'
        folder = 'figures'
        folder = os.path.join(root, folder)
        if not os.path.exists(folder):
            os.mkdir(folder)
        fname = 'field'
        fname = os.path.join(folder, fname)
        plt.savefig(fname + '.pdf')
    else:
        plt.show()


def plot_animation(simulation, agents, field, frames=None, save=False):
    # TODO: cmap
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set(xlim=field['x_dims'],
           ylim=field['y_dims'],
           xlabel=r'$ x $',
           ylabel=r'$ y $')

    line, = ax.plot([], [], marker='o', lw=0, alpha=0.5)

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
