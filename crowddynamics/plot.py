"""Visualization of simulation geometry"""
import os
from collections import Iterable
from contextlib import contextmanager

import bokeh.io
import bokeh.plotting
import matplotlib.ticker as plticker
from matplotlib import pyplot as plt, cm
from shapely.geometry import LineString, Point, Polygon, LinearRing


def plot_field(domain, obstacles, targets, **fig_kw):
    # TODO: polygon patch
    # TODO: default styles
    fig, ax = plt.subplots(**fig_kw)

    return fig, ax


def plot_navigation(mgrid, dmap, phi, dir_map=None, frequency=20, **fig_kw):
    """Plot distance map

    Args:
        mgrid (numpy.ndarray):
            Array created using numpy.meshgrid
        dmap (numpy.ndarray):
            Distance map. Plotted as countour.
        phi (numpy.ma.MaskedArray):
        dir_map (numpy.ndarray): 
            Direction map. Plotted as quiver.
        frequency (int): 
        **fig_kw:
            Key values for plt.subplots
    
    Returns:
        (Figure, Axes)
    
    Examples:
        >>> mgrid, dmap, phi = distance_map(...)
        >>> dir_map = direction_map(dmap)
        >>> fig, ax = plot_navigation(mgrid, dmap, phi, dir_map, 
        >>>                           figsize=(20, 20))
        >>> plt.show()
    """
    fig, ax = plt.subplots(**fig_kw)

    # This locator puts ticks at regular intervals
    loc = plticker.MultipleLocator(base=0.5)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    # Plot of distance map
    X, Y = mgrid
    bbox = (X.min(), X.max(), Y.min(), Y.max())
    ax.imshow(dmap, interpolation='bilinear', origin='lower', cmap=cm.gray,
              extent=bbox)
    ax.contour(X, Y, dmap, 30, linewidths=1, colors='gray')  # Contour lines
    ax.contour(X, Y, phi.mask, [0], linewidths=1, colors='black')  # Obstacles

    # Plot of direction map as quiver plot (vector field)
    if dir_map is not None:
        U, V = dir_map
        ax.quiver(X[::frequency, ::frequency],
                  Y[::frequency, ::frequency],
                  U[::frequency, ::frequency],
                  V[::frequency, ::frequency],
                  units='width')

    return fig, ax


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
    """Bokeh plot

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


def add_shape(fig, geom, *args, **kwargs):
    """Add Shapely geom into Bokeh plot.

    Args:
        fig (bokeh.plotting.figure.Figure):
        geom (shapely.geometry.BaseGeometry):
    """
    if isinstance(geom, Point):
        fig.circle(*geom.xy, *args, **kwargs)
    elif isinstance(geom, (LineString, LinearRing)):
        fig.line(geom.coords, *args, **kwargs)
    elif isinstance(geom, Polygon):
        fig.patch(*geom.exterior.xy, *args, **kwargs)
    elif isinstance(geom, Iterable):
        for item in geom:
            add_shape(fig, item, *args, **kwargs)
