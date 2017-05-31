import os
from contextlib import contextmanager

import bokeh.io
import bokeh.plotting
import numpy as np
from bokeh.models import Range1d
from bokeh.plotting import Figure
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry


# Bokeh

# TODO: implement bokeh interface with bokeh descriptors

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


def add_geom(fig: Figure, geom: BaseGeometry, **kwargs):
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
            add_geom(fig, item, **kwargs)
    else:
        raise TypeError('Object geom {geom} no instance of {types}.'.format(
            geom=geom, types=BaseGeometry))


def add_field(fig, field):
    """Plot Field
    
    Args:
        fig: 
        field: 
        
    """

    # Plot spawn and target indices
    add_geom(fig, field.domain, alpha=0.05)

    for spawn in field.spawns:
        add_geom(fig, spawn, alpha=0.5, line_width=0, color='green')

    for target in field.targets:
        add_geom(fig, target, alpha=0.5, line_dash='dashed', color='olive')

    add_geom(fig, field.obstacles, alpha=0.8)


def add_distance_map(fig, mgrid, distance_map, **kwargs):
    """Contour plot of distance from target
    
    http://bokeh.pydata.org/en/latest/docs/gallery/image.html
    
    Args:
        fig: 
        mgrid: 
        distance_map: 
        
    """
    kwargs.setdefault('palette', 'Greys256')
    minx, miny, maxx, maxy = mgrid.bounds
    fig.image([distance_map], minx, miny, maxx, maxy, **kwargs)


def add_direction_map(fig: Figure, mgrid, direction_map, freq=2, **kwargs):
    """Quiver plot of direction map
    
    http://bokeh.pydata.org/en/latest/docs/gallery/quiver.html
    
    Args:
        fig: 
        mgrid: 
        direction_map: 
        **kwargs: 
        
    """
    X, Y = mgrid.values
    U, V = direction_map

    x0 = X[::freq, ::freq].flatten()
    y0 = Y[::freq, ::freq].flatten()

    x1 = x0 + 0.9 * freq * mgrid.step * U[::freq, ::freq].flatten()
    y1 = y0 + 0.9 * freq * mgrid.step * V[::freq, ::freq].flatten()

    fig.segment(x0, y0, x1, y1, **kwargs)
    fig.triangle(x1, y1, size=4.0,
                 angle=np.arctan2(V[::freq, ::freq].flatten(),
                                  U[::freq, ::freq].flatten()) - np.pi / 2)


def plot_field(field, step=0.02, radius=0.3, strength=0.3, **kwargs):
    bokeh.io.output_file(field.name + '.html', field.name)
    p = bokeh.plotting.Figure(**kwargs)

    if field.domain:
        minx, miny, maxx, maxy = field.domain.bounds
    else:
        minx, miny, maxx, maxy = field.convex_hull().bounds

    set_aspect(p, (minx, maxx), (miny, maxy))
    p.grid.minor_grid_line_color = 'navy'
    p.grid.minor_grid_line_alpha = 0.05

    # indices = chain(range(len(self.targets)), ('closest',))
    # for index in indices:
    #     mgrid, distance_map, direction_map = \
    #         self.navigation_to_target(index, step, radius, strength)

    mgrid, distance_map, direction_map = field.navigation_to_target(
        'closest', step, radius, strength)

    # TODO: masked values on distance map
    add_distance_map(p, mgrid, distance_map.filled(1.0),
                     legend='distance_map')
    add_direction_map(p, mgrid, direction_map, legend='direction_map')

    add_geom(p, field.domain,
             legend='domain',
             alpha=0.05)

    for i, spawn in enumerate(field.spawns):
        add_geom(p, spawn,
                 legend='spawn_{}'.format(i),
                 alpha=0.5,
                 line_width=0,
                 color='green',)

    for i, target in enumerate(field.targets):
        add_geom(p, target,
                 legend='target_{}'.format(i),
                 alpha=0.8,
                 line_width=3.0,
                 line_dash='dashed',
                 color='olive',)

    add_geom(p, field.obstacles,
             legend='obstacles',
             line_width=3.0,
             alpha=0.8, )

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    bokeh.io.show(p)


# Anytree

def render_tree(tree, filepath):
    """Wrapper around rendering trees into `png` or `dot` formats. Tree are 
    rendered from the root node.
    
    Args:
        tree (NodeMixin): 
        filepath (str|Path): 
    """
    from anytree.dotexport import RenderTreeGraph
    tree = RenderTreeGraph(tree.root)
    base, ext = os.path.splitext(filepath)
    if ext == '.png':
        tree.to_picture(filename=filepath)
    elif ext == '.dot':
        tree.to_dotfile(filename=filepath)
    else:
        raise Exception
