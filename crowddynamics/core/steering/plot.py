from matplotlib import pyplot as plt, cm


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
