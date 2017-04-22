"""Visualization of simulation geometry"""
import os
from contextlib import contextmanager

import bokeh.io
import bokeh.plotting
from bokeh.plotting.figure import Figure
from crowddynamics.core.geometry import geom_to_mpl
from matplotlib import cm
from matplotlib.patches import PathPatch
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseMultipartGeometry, BaseGeometry


# Matplotlib


def plot_field(fig, ax, domain, obstacles, targets):
    """Plot field"""
    # TODO: polygon patch
    # TODO: default styles
    if domain:
        ax.add_patch(PathPatch(
            geom_to_mpl(domain),
            # color='0.25', linewidth=0.0, alpha=0.8, fill=True
        ))

    if obstacles:
        ax.add_patch(PathPatch(
            geom_to_mpl(obstacles),
            # color='1.0', fill=True
        ))

    if targets:
        ax.add_patch(PathPatch(
            geom_to_mpl(targets),
            color='0.75', linestyle='--', alpha=0.5
        ))

    return fig, ax


def plot_navigation(fig, ax, mgrid, dmap, dir_map=None, frequency=20):
    """Plot distance map

    Args:
        fig: Matplotlib figure instance
        ax: Matplotlib Axes instance
        mgrid:
            Array created using numpy.meshgrid
        dmap:
            Distance map. Plotted as countour.
        dir_map: 
            Direction map. Plotted as quiver.
        frequency (int): 
        **fig_kw:
            Key values for plt.subplots
    
    Returns:
        (Figure, Axes)
    """
    X, Y = mgrid
    bbox = (X.min(), X.max(), Y.min(), Y.max())
    ax.imshow(dmap, interpolation='bilinear', origin='lower', cmap=cm.gray,
              extent=bbox)
    ax.contour(X, Y, dmap, 30, linewidths=1, colors='gray')  # Contour lines
    if hasattr(dmap, 'mask'):
        ax.contour(X, Y, dmap.mask, [0], linewidths=1, colors='black')  # Obstacles

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


# Bokeh


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


def geom_to_bokeh(fig: Figure,
                  geom: BaseGeometry,
                  *args, **kwargs):
    """Add Shapely geom into Bokeh plot.

    Args:
        fig (Figure):
        geom (BaseGeometry):
    """
    if isinstance(geom, Point):
        fig.circle(geom.coords, *args, **kwargs)
    elif isinstance(geom, LineString):
        fig.line(geom.coords, *args, **kwargs)
    elif isinstance(geom, Polygon):
        fig.patch(geom.exterior.coords, *args, **kwargs)
    elif isinstance(geom, BaseMultipartGeometry):
        for item in geom:
            geom_to_bokeh(fig, item, *args, **kwargs)
    else:
        raise TypeError
