import logging
from collections import Iterable

import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Polygon

from crowddynamics.core.geometry import shapes_to_point_pairs

try:
    import skfmm
    import skimage.draw
except ImportError:
    raise Warning("Navigation algorithm cannot be used if scikit-fmm or "
                  "scikit-image are not installed")

from .vector2D import angle_nx2


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


class ExitSelection:
    """Exit selection policy."""

    def __init__(self, simulation):
        self.simulation = simulation


class NavigationMap(object):
    initial = -1.0
    target = 1.0
    obstacle = True

    def __init__(self, domain, step=0.01):
        self.domain = domain
        self.step = step

        minx, miny, maxx, maxy = domain.bounds  # Bounding box
        self.grid = np.meshgrid(np.arange(minx, maxx + step, step=step),
                                np.arange(miny, maxy + step, step=step), )

    def points_to_indices(self, points):
        return np.round(points / self.step).astype(int)

    def set_values(self, shape, array, value):
        if isinstance(shape, Polygon):
            points = np.asarray(shape.exterior)
            points = self.points_to_indices(points)
            x, y = points[:, 0], points[:, 1]
            j, i = skimage.draw.polygon(x, y)
            array[i, j] = value
        elif isinstance(shape, LineString):
            points = shapes_to_point_pairs(shape)
            points = self.points_to_indices(points)
            for args in points:
                j, i = skimage.draw.line(*args.flatten())
                array[i, j] = value
        elif isinstance(shape, Iterable):
            for shape_ in shape:
                self.set_values(shape_, array, value)
        else:
            raise Exception()

    def distance_map(self, obstacles, targets):
        contour = np.full_like(self.grid[0], self.initial, dtype=np.float64)
        self.set_values(targets, contour, self.target)

        mask = np.full_like(self.grid[0], False, dtype=np.bool_)
        self.set_values(obstacles, mask, self.obstacle)

        contour = np.ma.MaskedArray(contour, mask)
        dmap = skfmm.distance(contour, dx=self.step)
        return dmap, contour

    def travel_time_map(self):
        pass

    def static(self):
        pass

    def dynamic(self, obstacles, targets, dynamic):
        pass


class Navigation(NavigationMap):
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

    # TODO: take into account radius of the agents

    def __init__(self, simulation):
        super().__init__(simulation.domain)

        self.simulation = simulation
        self.dist_map = None
        self.direction_map = None

    def static_potential(self):
        logging.info("")

        if self.dist_map is None:
            self.dist_map, contour = self.distance_map(
                self.simulation.obstacles,
                self.simulation.exits,
            )

        u, v = np.gradient(self.dist_map)
        l = np.hypot(u, v)  # Normalize
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
        indices = self.points_to_indices(points)
        indices = np.fliplr(indices)
        # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        # TODO: Handle index out of bounds -> numpy.take
        d = self.direction_map[indices[:, 0], indices[:, 1], :]
        self.simulation.agent.target_direction[i] = d


class Orientation:
    """
    Target orientation
    """

    def __init__(self, simulation):
        self.simulation = simulation

    def update(self):
        if self.simulation.agent.orientable:
            dir_to_orient = angle_nx2(self.simulation.agent.target_direction)
            self.simulation.agent.target_angle[:] = dir_to_orient
