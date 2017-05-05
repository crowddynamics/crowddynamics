import os
from contextlib import contextmanager

import bokeh.io
import bokeh.plotting
from bokeh.models import Range1d
from bokeh.plotting.figure import Figure
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseMultipartGeometry, BaseGeometry


def set_aspect(fig, x, y, aspect=1, margin=0.1):
    """Set the plot ranges to achieve a given aspect ratio.

    https://stackoverflow.com/questions/26674779/bokeh-plot-with-equal-axes

    Args:
      fig (bokeh Figure): The figure object to modify.
      x (iterable): The x-coordinates of the displayed data.
      y (iterable): The y-coordinates of the displayed data.
      aspect (float, optional): The desired aspect ratio. Defaults to 1.
          Values larger than 1 mean the plot is squeezed horizontally.
      margin (float, optional): The margin to add for glyphs (as a fraction
          of the total plot range). Defaults to 0.1
    """
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)

    width = (xmax - xmin) * (1 + 2 * margin)
    width = 1.0 if width <= 0 else width

    height = (ymax - ymin) * (1 + 2 * margin)
    height = 1.0 if height <= 0 else height

    r = aspect * (fig.plot_width / fig.plot_height)

    if width < r * height:
        width = r * height
    else:
        height = width / r

    xcenter = 0.5 * (xmax + xmin)
    ycenter = 0.5 * (ymax + ymin)

    fig.x_range = Range1d(xcenter - 0.5 * width, xcenter + 0.5 * width)
    fig.y_range = Range1d(ycenter - 0.5 * height, ycenter + 0.5 * height)


@contextmanager
def figure(filename, show=False, save=False, **kwargs):
    """Context manager for using :class:`bokeh.plotting.figure`.

    Args:
        filename (str):
        show (bool):
        save (bool):

    Yields:
        Figure:
    
    Examples:
        >>> with figure('figure.html', show=True, save=False) as p:
        >>>     p.patch(...)
    """
    base, ext = os.path.splitext(filename)
    _, name = os.path.split(base)
    if ext != '.html':
        filename += '.html'

    bokeh.io.output_file(filename, name)
    fig = bokeh.plotting.Figure(**kwargs)

    yield fig

    if show:
        bokeh.io.show(fig)

    if save:
        bokeh.io.save(fig)


def plot_geom(fig: Figure, geom: BaseGeometry, **kwargs):
    """Add Shapely geom into Bokeh plot.

    Args:
        fig (Figure):
        geom (BaseGeometry):
    """
    if isinstance(geom, Point):
        fig.circle(*geom.xy, **kwargs)
    elif isinstance(geom, LineString):
        fig.line(*geom.xy, **kwargs)
    elif isinstance(geom, Polygon):
        fig.patch(*geom.exterior.xy, **kwargs)
    elif isinstance(geom, BaseMultipartGeometry):
        for item in geom:
            plot_geom(fig, item, **kwargs)
    else:
        raise TypeError('Object geom {geom} no instance of {types}.'.format(
            geom=geom, types=BaseGeometry))


def plot_field(fig, field):
    """Plot Field
    
    Args:
        fig: 
        field: 
        
    """

    # Plot spawn and target indices
    plot_geom(fig, field.domain, alpha=0.05)

    for spawn in field.spawns:
        plot_geom(fig, spawn, alpha=0.5, line_width=0, color='green')

    for target in field.targets:
        plot_geom(fig, target, alpha=0.5, line_dash='dashed', color='olive')

    plot_geom(fig, field.obstacles, alpha=0.8)


def plot_distance_map(fig, mgrid, distance_map, **kwargs):
    """Contour plot of distance from target
    
    http://bokeh.pydata.org/en/latest/docs/gallery/image.html
    
    Args:
        fig: 
        mgrid: 
        distance_map: 
        
    """
    kwargs.setdefault('palette', 'Greys256')

    # TODO: transform masked values to transparent
    # distance_map.filled(...)
    minx, miny, maxx, maxy = mgrid.bounds
    fig.image([distance_map], minx, miny, maxx, maxy, **kwargs)


def plot_direction_map(fig, mgrid, direction_map, freq=2, **kwargs):
    """Quiver plot of direction map
    
    http://bokeh.pydata.org/en/latest/docs/gallery/quiver.html
    
    Args:
        fig: 
        mgrid: 
        direction_map: 
        **kwargs: 
        
    """
    # TODO: add arrow heads

    X, Y = mgrid.values
    U, V = direction_map

    x0 = X[::freq, ::freq].flatten()
    y0 = Y[::freq, ::freq].flatten()

    x1 = x0 + 0.9 * freq * mgrid.step * U[::freq, ::freq].flatten()
    y1 = y0 + 0.9 * freq * mgrid.step * V[::freq, ::freq].flatten()

    fig.segment(x0, y0, x1, y1, **kwargs)
