import logging
import numpy as np

from src.multiagent.simulation import Configuration
from .vector2D import angle_nx2


class ExitSelection:
    """Exit selection policy."""
    def __init__(self, simulation: Configuration):
        self.simulation = simulation


class Navigation:
    """
    Target direction. Algorithm based on solving the continous shortest path
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
    def __init__(self, simulation: Configuration):
        self.simulation = simulation

    def distance_map(self, step=0.01):
        """
        Computes distance map for the simulation domain.

        * From rectangular grid from the bounding box of the polygonal domain
        * Set initial value of the grid to -1
        * Discretize linestring of obstacles and exits using *Bresenham's line algorithm* [1]_.
        * Set values of points that contain exit to 1
        * Mask points that contain obstacle

        .. [1] https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

        :param step: Meshgrid cell size (width, height) in meters.
        :return:
        """
        logging.info("")

        import skfmm

        # Discretize the domain into a grid.
        x, y = self.simulation.domain.exterior.xy
        x, y = np.asarray(x), np.asarray(y)

        # Bounding box of domain
        lim = (x.min(), x.max()), (y.min(), y.max())
        diff = np.array([np.diff(lim[0]), np.diff(lim[1])])
        n = (diff / step).astype(int)
        dx = diff / n
        grid = np.meshgrid(np.linspace(*lim[0], num=n[0]),
                           np.linspace(*lim[1], num=n[1]))
        values = grid[0].flatten(), grid[1].flatten()

        # Indices of exits and obstacles.
        # Bresenham's line algorithm
        indices_exit = None
        indices_obstacles = None

        # Set contour.
        # Exits. Zero contour defines the exit.
        # Obstacles are defined by masked values.
        phi = np.zeros_like(grid[0])
        phi[:] = -1
        phi[indices_exit] = 1
        mask = np.zeros_like(phi, dtype=bool)
        mask[indices_obstacles] = True
        phi = np.ma.MaskedArray(phi, mask)

        dist_map = skfmm.distance(phi, dx=dx)

        return

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
    def __init__(self, simulation: Configuration):
        self.simulation = simulation

    def update(self):
        self.simulation.agent.target_angle[:] = angle_nx2(
            self.simulation.agent.target_direction)
