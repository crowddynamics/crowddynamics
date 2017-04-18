"""Visualization of simulation geometry"""
import os
from contextlib import contextmanager

import bokeh.io
import bokeh.plotting
import matplotlib.ticker as plticker
from bokeh.plotting.figure import Figure
from matplotlib import pyplot as plt, cm
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseMultipartGeometry, BaseGeometry


# Matplotlib


def plot_field(domain, obstacles, targets, **fig_kw):
    # TODO: polygon patch
    # TODO: default styles
    fig, ax = plt.subplots(**fig_kw)

    return fig, ax


def plot_navigation(mgrid, dmap, dir_map=None, frequency=20, **fig_kw):
    """Plot distance map

    Args:
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
    
    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from shapely.geometry import LineString, Polygon
        >>> 
        >>> from crowddynamics.core.steering.navigation import distance_map, \
        >>>     direction_map, meshgrid
        >>> from crowddynamics.plot import plot_navigation
        >>> 
        >>> step = 0.01
        >>> height = 3.0
        >>> width = 4.0
        >>> radius = 0.3
        >>> 
        >>> domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
        >>> targets = LineString([(0, height / 2), (0, height)])
        >>> obstacles = LineString([(0, height / 2), (width * 3 / 4, height / 2)]) | \
        >>>             domain.exterior - targets
        >>> 
        >>> mgrid = meshgrid(step, *domain.bounds)
        >>> dmap_exits = distance_map(mgrid, targets,
        >>>                           obstacles.buffer(radius).intersection(domain))
        >>> dmap_obs = distance_map(mgrid, obstacles, None)
        >>> dir_map_exits = direction_map(dmap_exits)
        >>> dir_map_obs = direction_map(dmap_obs)
        >>> 
        >>> plot_navigation(mgrid.values, dmap_exits, dir_map_exits, frequency=5, figsize=(16, 16))
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
