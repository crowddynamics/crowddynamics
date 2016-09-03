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

        self.grid = None
        self.dmap = None

    @staticmethod
    def _dicretize_polygon_to_grid(polygon, step):
        # Discretize the domain into a grid by its bounding box.
        # TODO: mask areas not in polygon
        x, y = polygon.exterior.xy
        x, y = np.asarray(x), np.asarray(y)
        grid = np.meshgrid(np.arange(x.min(), x.max() + step, step=step),
                           np.arange(y.min(), y.max() + step, step=step), )
        return grid

    @staticmethod
    def _linestrings_to_points(linestrings):
        points = []
        for linestring in linestrings:
            a = np.asarray(linestring)
            for i in range(len(a) - 1):
                points.append(a[i:i + 2])
        ret = np.array(points)
        return ret

    @staticmethod
    def _set_line_value(points, out, value):
        for args in points:
            # TODO: Check correctness
            i, j = skimage.draw.line(*args.flatten())
            out[j, i] = value

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

        target = self.simulation.exits
        if not target:
            target = [self.simulation.domain.exterior]

        # Discretize the domain into a grid by its bounding box.
        grid = self._dicretize_polygon_to_grid(self.simulation.domain, step)
        points = self._linestrings_to_points(target)
        points2 = self._linestrings_to_points(self.simulation.obstacles)

        # FIXME
        # Indices of the nearest points in the grid
        indices_exits = np.round(points / step).astype(int)
        indices_obstacle = np.round(points2 / step).astype(int)

        # Set of exits and obstacles with line drawing algorithm
        phi = np.zeros_like(grid[0])  # Contour
        obstacles = np.zeros_like(grid[0], dtype=bool)  # Obstacles
        phi[:] = -1                   # Initial values
        value_target = 1                    # Values to be set to be target
        value_obstacle = True               # Value indicating an obstacle

        self._set_line_value(indices_exits, phi, value_target)
        self._set_line_value(indices_obstacle, obstacles, value_obstacle)

        phi = np.ma.MaskedArray(phi, obstacles)
        dmap = skfmm.distance(phi, dx=step)

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        X, Y = grid
        plt.figure(figsize=(12, 12))
        plt.title('Distance map from exit.')
        im = plt.imshow(dmap,
                        interpolation='bilinear',
                        origin='lower',
                        cmap=cm.gray,
                        extent=(X.min(), X.max(), Y.min(), Y.max()))
        plt.contour(X, Y, dmap, 30, linewidths=1, colors='gray')
        plt.contour(X, Y, phi.mask, [0], linewidths=1, colors='black')
        plt.savefig("{}.pdf".format(self.simulation.name))

        self.grid = grid
        self.dmap = dmap

    def static_potential(self):
        logging.info("")
        pass

    def dynamic_potential(self):
        logging.info("")
        pass

    def update(self):
        # self.agent.target_direction = None
        pass


class Orientation:
    """
    Target orientation
    """

    def __init__(self, simulation):
        self.simulation = simulation

    def update(self):
        self.simulation.agent.target_angle[:] = angle_nx2(
            self.simulation.agent.target_direction)
