import numba
import numpy as np
from scipy.spatial import Voronoi
from numba import f8, i8

from crowddynamics.core.geom2D import polygon_area


def density_classical(points, cell_size):
    r"""Classical definition of density defined the density :math:`D` as number 
    of agents per unit of area.

    .. math::
       D = \frac{N}{|A|}\quad \left[\mathrm{\frac{1}{m^{2}}}\right]
    
    where
    
    - :math:`N` is the number of agents
    - :math:`|A|` is the size of the area

    However the classical definition of density does not work as well when 
    measuring density of crowd or granular media as it does for measuring 
    density of fluids which have :math:`> 10^{18}` particles per 
    :math:`\mathrm{mm}^3`.
    """
    indices = np.floor(points / cell_size)
    x, y = indices[:, 0], indices[:, 1]
    x_min, y_min = x.min(), y.min()
    shape = x.max() - x_min, y.max() - y_min
    count = np.zeros(shape, dtype=np.int64)
    for i, j in zip(x, y):
        count[i - x_min, j - y_min] += 1
    return count / cell_size ** 2


def _definition1(points, cell_size, area, point_region):
    pass


@numba.jit([f8[:, :](f8[:, :], f8, f8[:], i8[:])],
           nopython=True, nogil=True, cache=True)
def _definition2(points, cell_size, area, point_region):
    # FIXME
    indices = (points / cell_size).astype(np.int64)
    x, y = indices[:, 0], indices[:, 1]
    x_min, y_min = x.min(), y.min()
    shape = x.max() - x_min + 1, y.max() - y_min + 1
    density = np.zeros(shape)
    area_sum = np.zeros(shape)

    for k, (i, j) in enumerate(zip(x, y)):
        index = i - x_min, j - y_min
        density[index] += 1
        area_sum[index] += area[point_region[k]]

    n, m = density.shape
    for i in range(n):
        for j in range(m):
            s = area_sum[i, j]
            if s > 0:
                density[i, j] /= s

    return density


def density_voronoi(points, cell_size, definition=2):
    r"""Density definition base on Voronoi-diagram. [Steffen2010]_
    
    Compute Voronoi-diagram for each position :math:`\mathbf{p}_i` giving cells 
    :math:`A_i` to each agent :math:`i`. Area of Voronoi cell is denoted 
    :math:`|A_i|`. Area :math:`A` is the area inside which we measure the 
    density.  

    Definition 1
        :math:`S = \{i : \mathbf{q} \in A_i \wedge \mathbf{q} \in A\}` is the 
        set of agents whose Voronoi-diagram has points that belong to area 
        :math:`A`.
        
        .. math::
            p_i(\mathbf{x}) =
            \frac{1}{|A_i|}
            \begin{cases}
                1 & \mathbf{x} \in A_i \\ 
                0 & \text{otherwise} 
            \end{cases}
        
        .. math::
            p(\mathbf{x}) = \sum_{i \in S} p_i(\mathbf{x})
        
        .. math::
            D_V &= \frac{1}{|A|} \int_A p(\mathbf{x}) d\mathbf{x} \\ 
                &= \frac{1}{|A|} \int_A \sum_{i \in S} p_i(\mathbf{x}) d\mathbf{x} \\
                &= \frac{1}{|A|} \sum_{i \in S} \int_A  p_i(\mathbf{x}) d\mathbf{x} \\
                &= \frac{1}{|A|} \sum_{i \in S} \frac{1}{|A_i|} \int_A  
                   \begin{cases}
                       1 & \mathbf{x} \in A_i \\ 
                       0 & \text{otherwise} 
                   \end{cases}
                   d\mathbf{x} \\
                &= \frac{1}{|A|} \sum_{i \in S} \frac{|A \cap A_i|}{|A_i|} 

    Definition 2
        :math:`S = \{i : \mathbf{p}_i \in A\}` is the set of agents and 
        :math:`N = |S|` the number of agents that is inside area :math:`A`.
    
        .. math::
            D_{V^\prime} = \frac{N}{\sum_{i \in S} |A_i|}

    Args: 
        points (numpy.ndarray):
            Two dimensional points :math:`\mathbf{p}_{i \in \mathbb{N}}`
        cell_size (float):
            Cell size of the meshgrid. Each cell represents an area :math:`A` 
            inside which density is measured.
        definition (int):
            Integer ``{1, 2}`` denoting which definition of density to use.

    Returns:
        numpy.ndarray: Grid of values density values for each cell. 

    """
    vor = Voronoi(points)

    # Compute areas for all of the Voronoi regions
    area = np.array([polygon_area(vor.vertices[i]) for i in vor.regions])

    # Maps point indices into Voronoi region indices

    if definition == 1:
        raise NotImplementedError
    elif definition == 2:
        return _definition2(points, cell_size, area, vor.point_region)
    else:
        raise ValueError
