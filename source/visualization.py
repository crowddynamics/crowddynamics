from collections import Iterable, deque
from itertools import repeat, islice

import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Arrow, Circle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation as animation
from matplotlib.collections import CircleCollection
import matplotlib.transforms as transforms

from source.io.path import default_path


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
    patch = lambda *args: Circle(*args, **kwargs)
    patches = map(patch, x, r)
    return patches


def linear_wall_patches(p, **kwargs):
    f = lambda arg: tuple(zip(*arg))
    data = tuple(map(f, p))
    patch = lambda args: Line2D(*args, **kwargs)
    patches = map(patch, data)
    return patches


def vector_patches(x, f, tol=0.0001, **kwargs):
    mag = np.hypot(f[:, 0], f[:, 1])
    mask = mag > tol
    x = x[mask]
    f = f[mask]
    patch = lambda *args: Arrow(*args, **kwargs)
    patches = map(patch, x[:, 0], x[:, 1], f[:, 0], f[:, 1])
    return patches


def add_patches(ax, patches):
    consume(map(ax.add_artist, patches))


def plot_field(position, radius, x_dims, y_dims,
               linear_wall=None, force=None, save=True):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set(xlim=x_dims, ylim=y_dims, xlabel=r'$ x $', ylabel=r'$ y $')

    add_patches(ax, agent_patches(position, radius, alpha=0.5))

    # if linear_wall is not None:
    #     add_patches(ax, linear_wall_patches(linear_wall, color='black'))
    #
    # if force is not None:
    #     add_patches(ax, vector_patches(position, force, alpha=0.8, width=0.15))

    if save:
        fname = default_path('field.pdf', 'documentation', 'figures')
        plt.savefig(fname)
    else:
        plt.show()


def plot_animation(simulation, agents, wall_params, field, frames=None,
                   save=False):
    """
    http://matplotlib.org/1.4.1/examples/animation/index.html
    http://matplotlib.org/examples/api/patch_collection.html
    """
    try:
        import seaborn
        seaborn.set()
    except ImportError():
        pass
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set(xlim=field['x_dims'],
           ylim=field['y_dims'],
           xlabel=r'$ x $',
           ylabel=r'$ y $')

    line, = ax.plot([], [], marker='o', lw=0, alpha=0.5)
    scatter = ax.scatter([], [])

    def init_lines():
        positions = agents['position']
        line.set_data(positions.T)
        linear_wall = wall_params['linear_params']
        if linear_wall is not None:
            add_patches(ax, linear_wall_patches(linear_wall, color='black'))
        return line,

    def update_lines(i):
        simu = next(simulation)
        line.set_data(simu['position'].T)
        return line,

    anim = animation.FuncAnimation(fig, update_lines,
                                   init_func=init_lines,
                                   frames=frames,
                                   interval=1,
                                   blit=True)

    if save:
        writer = animation.FFMpegWriter(fps=30, bitrate=1800)
        fname = default_path('anim.mkv', 'documentation', 'animations')
        anim.save(fname, writer=writer)
    else:
        plt.show()
