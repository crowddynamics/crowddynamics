from matplotlib import cm


def plot_navigation(fig, ax, mgrid, dmap, dir_map=None, frequency=20):
    """Plot distance map

    Args:
        fig: Matplotlib figure instance
        ax: Matplotlib Axes instance
        mgrid: Array created using numpy.meshgrid
        dmap: Distance map. Plotted as countour.
        dir_map: Direction map. Plotted as quiver.
        frequency (int): 
        **fig_kw: Key values for plt.subplots
    
    Returns:
        (Figure, Axes)
    """
    X, Y = mgrid
    minx, maxx, miny, maxy = X.min(), X.max(), Y.min(), Y.max()

    ax.imshow(dmap, interpolation='bilinear', origin='lower', cmap=cm.gray,
              extent=(minx, maxx, miny, maxy))
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
