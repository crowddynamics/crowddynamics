import logging

import numpy as np

try:
    import skfmm
    import skimage.draw
except ImportError:
    print(Warning("Navigation algorithm cannot be used if scikit-fmm or "
                  "scikit-image are not installed"))

from .vector2D import angle_nx2


class ExitSelection:
    """Exit selection policy."""

    def __init__(self, simulation):
        self.simulation = simulation


def plot_dmap(grid, dmap, phi, name):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    X, Y = grid
    plt.figure(figsize=(12, 12))
    plt.title('Distance map from exit.')
    plt.imshow(dmap, interpolation='bilinear', origin='lower', cmap=cm.gray,
               extent=(X.min(), X.max(), Y.min(), Y.max()))
    plt.contour(X, Y, dmap, 30, linewidths=1, colors='gray')
    plt.contour(X, Y, phi.mask, [0], linewidths=1, colors='black')
    plt.savefig("distance_map_{}.pdf".format(name))


def _dicretize_polygon_to_grid(polygon, step):
    # Discretize the domain into a grid by its bounding box.
    # TODO: mask areas not in polygon
    x, y = polygon.exterior.xy
    x, y = np.asarray(x), np.asarray(y)
    grid = np.meshgrid(np.arange(x.min(), x.max() + step, step=step),
                       np.arange(y.min(), y.max() + step, step=step), )
    return grid


def _linestrings_to_points(linestrings):
    points = []
    for linestring in linestrings:
        a = np.asarray(linestring)
        for i in range(len(a) - 1):
            points.append(a[i:i + 2])
    ret = np.array(points)
    return ret


def _set_line_value(points, out, value):
    for args in points:
        # points are y, x order -> flip to row, col order
        j, i = skimage.draw.line(*args.flatten())
        out[i, j] = value


def points_to_indices(points, step):
    indices = np.round(points / step).astype(int)
    return indices


def distance_map(domain, exits=None, obstacles=None, step=0.01):
    target = exits
    if not target:
        target = [domain.exterior]

    if not obstacles:
        obstacles = ()

    # Discretize the domain into a grid by its bounding box.
    grid = _dicretize_polygon_to_grid(domain, step)
    points_target = _linestrings_to_points(target)
    obstacle_points = _linestrings_to_points(obstacles)

    # Indices of the nearest points in the grid
    indices_exits = points_to_indices(points_target, step)
    indices_obstacle = points_to_indices(obstacle_points, step)

    # Set of exits and obstacles with line drawing algorithm
    phi = np.zeros_like(grid[0])  # Contour
    obstacles = np.zeros_like(grid[0], dtype=bool)  # Obstacles
    phi[:] = -1  # Initial values
    value_target = 1  # Values to be set to be target
    value_obstacle = True  # Value indicating an obstacle

    _set_line_value(indices_exits, phi, value_target)
    _set_line_value(indices_obstacle, obstacles, value_obstacle)

    phi = np.ma.MaskedArray(phi, obstacles)
    dmap = skfmm.distance(phi, dx=step)

    return grid, dmap, phi


def travel_time_map():
    pass


class Navigation:
    """Determining target direction of an agent in multi-agent simulation.

    Algorithm based on solving the continous shortest path
    problem by solving eikonal equation. [1]_, [2]_

    There are at least two open source eikonal solvers. Fast marching method
    (FMM) [3]_ for rectangular and tetrahedral meshes using Python and C++ and
    fast iterative method (FIM) [4]_ for triangular meshes using c++ and CUDA.

    In this implementation we use the FMM algorithm because it is simpler.

    .. [1] Kretz, T., Große, A., Hengst, S., Kautzsch, L., Pohlmann, A., & Vortisch, P. (2011). Quickest Paths in Simulations of Pedestrians. Advances in Complex Systems, 14(5), 733–759. http://doi.org/10.1142/S0219525911003281
    .. [2] Cristiani, E., & Peri, D. (2015). Handling obstacles in pedestrian simulations: Models and optimization. Retrieved from http://arxiv.org/abs/1512.08528
    .. [3] https://github.com/scikit-fmm/scikit-fmm
    .. [4] https://github.com/SCIInstitute/SCI-Solver_Eikonal
    """

    def __init__(self, simulation):
        self.simulation = simulation

        self.step = None
        self.grid = None
        self.dmap = None

        self.direction_map = None

    def distance_map(self, step=0.01):
        """Computes distance map for the simulation domain.

        * From rectangular grid from the bounding box of the polygonal domain
        * Set initial value of the grid to -1
        * Discretize linestring of obstacles and exits using line drawing algorithm
        * Set values of points that contain exit to 1
        * Mask points that contain obstacle

        :param step: Meshgrid cell size (width, height) in meters.
        :return:
        """
        logging.info("")

        grid, dmap, phi = distance_map(self.simulation.domain,
                                       self.simulation.exits,
                                       self.simulation.obstacles,
                                       step=step)
        self.step = step
        self.grid = grid
        self.dmap = dmap

    def static_potential(self):
        logging.info("")

        if self.dmap is None:
            self.distance_map()
        u, v = np.gradient(self.dmap)
        l = np.hypot(u, v)   # Normalize
        direction = np.zeros(u.shape + (2,))
        # Flip order from (row, col) to (x, y)
        direction[:, :, 0] = v / l
        direction[:, :, 1] = u / l
        self.direction_map = direction

    def distance_map_agents(self):
        pass

    def dynamic_potential(self):
        logging.info("")
        raise NotImplementedError

    def update(self):
        i = self.simulation.agent.indices()
        points = self.simulation.agent.position[i]
        indices = points_to_indices(points, self.step)
        indices = np.fliplr(indices)
        # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        # TODO: Handle index out of bounds
        d = self.direction_map[indices[:, 0], indices[:, 1], :]
        self.simulation.agent.target_direction[i] = d


class Orientation:
    """
    Target orientation
    """

    def __init__(self, simulation):
        self.simulation = simulation

    def update(self):
        self.simulation.agent.target_angle[:] = angle_nx2(
            self.simulation.agent.target_direction)
