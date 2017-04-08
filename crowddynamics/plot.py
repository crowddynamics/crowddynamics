import os
from collections import Iterable
from contextlib import contextmanager

import bokeh.io
import bokeh.plotting
import numpy as np
from matplotlib import pyplot as plt, cm
from shapely.geometry import LineString, Point, Polygon, LinearRing


def plot_distance_map(mgrid, dmap, phi):
    """
    Plot distance map

    Args:
        mgrid (numpy.meshgrid):
        dmap (numpy.ndarray):
        phi (numpy.ma.MaskedArray):
    """
    X, Y = mgrid
    bbox = (X.min(), X.max(), Y.min(), Y.max())

    opts = dict(
        figsize=(12, 12),
    )

    fig, ax = plt.subplots(**opts)

    # Distance map plot
    ax.imshow(dmap, interpolation='bilinear', origin='lower', cmap=cm.gray,
              extent=bbox)
    ax.contour(X, Y, dmap, 30, linewidths=1, colors='gray')  # Contour lines
    ax.contour(X, Y, phi.mask, [0], linewidths=1, colors='black')  # Obstacles

    # plt.savefig("distance_map_{}.pdf".format(name))
    ax.show()


def path(filename, save_dir="output",
         base=os.path.dirname(os.path.abspath(__file__))):
    # TODO: slugify filename
    # TODO: filename existing
    d = os.path.join(base, save_dir)
    os.makedirs(d, exist_ok=True)
    filename = filename.replace(" ", "_")
    return os.path.join(d, filename)


@contextmanager
def figure(name, show=False, save=False):
    """
    Plot polygons

    Args:
        name (str):
        show (bool):
        save (bool):

    Yields:
        bokeh.plotting.figure.Figure:
    """
    # Metadata
    filename = name + ".html"
    title = name.replace("_", "").capitalize()

    # Bokeh figure
    bokeh.io.output_file(path(filename), title)
    p = bokeh.plotting.figure()
    yield p

    if show:
        bokeh.io.show(p)

    if save:
        bokeh.io.save(p)


def add_shape(fig, shape, *args, **kwargs):
    """
    Add Shapely ``shape`` into ``Bokeh`` plot.

    Args:
        fig (bokeh.plotting.figure.Figure):
        shape (shapely.geometry.BaseGeometry):
    """
    if isinstance(shape, Point):
        x, y = shape.xy
        fig.circle(x, y, *args, **kwargs)
    elif isinstance(shape, (LineString, LinearRing)):
        coords = np.asarray(shape.coords)
        fig.line(coords, *args, **kwargs)
    elif isinstance(shape, Polygon):
        values = np.asarray(shape.exterior)
        fig.patch(values[:, 0], values[:, 1], *args, **kwargs)
    elif isinstance(shape, Iterable):
        for item in shape:
            add_shape(fig, item, *args, **kwargs)
