"""
Spatial partitioning algorithms.

- BlockList
- ConvexHull

Since crowd simulations are only dependent on interactions with agents close by
we can partition the space into smaller chunk in order to avoid having to loop
with agents far a away.
"""
import numpy as np
import numba
from numba import f8, i8


@numba.jit(nopython=True)
def block_list(points, cell_size):
    """
    Block list

    Args:
        points (numpy.ndarray):
            Array of ``shape=(size, 2)`` to be block listed.

        cell_size (float):
            Positive real number. Width and height of the rectangular mesh.

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray):
            - ``index_list``
            - ``count``
            - ``offset``
            - ``x_min``
            - ``x_max``

    """
    assert cell_size > 0
    assert points.ndim == 2
    assert points.shape[1] == 2

    # Dimensions (rows, columns)
    n, m = points.shape

    # Compute index ranges for indices
    x_min = (points[0, :] / cell_size).astype(np.int64)
    x_max = (points[0, :] / cell_size).astype(np.int64)
    for i in range(1, n):
        for j in range(m):
            x = np.int64(points[i, j] / cell_size)
            if x < x_min[j]:
                x_min[j] = x
            if x > x_max[j]:
                x_max[j] = x

    # Blocks
    indices = np.zeros(shape=points.shape, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            x = np.int64(points[i, j] / cell_size) - x_min[j]
            indices[i, j] = x

    #
    x_max = x_max - x_min

    # TODO: Implementation for sparse block lists.
    #       Maximum amount of blocks == len(points)

    # Count how many points go into each point
    size = np.prod(x_max + 1)
    count = np.zeros(size, dtype=np.int64)
    for i in range(n):
        index = indices[i, 0]
        for j in range(1, m):
            index *= x_max[j] + 1
            index += indices[i, j]
        count[index] += 1

    # Index list
    index_list = np.zeros(points.shape[0], dtype=np.int64)
    offset = count.cumsum() - 1  # Offset indices
    for i in range(n):
        index = indices[i, 0]
        for j in range(1, m):
            index *= x_max[j] + 1
            index += indices[i, j]
        index_list[offset[index]] = i
        offset[index] -= 1

    offset += 1

    return index_list, count, offset, x_min, x_max


spec = (
    ("cell_width", f8),
    ("index_list", i8[:]),
    ("count", i8[:]),
    ("offset", i8[:]),
    ("x_min", i8[:]),
    ("x_max", i8[:]),
    ("shape", i8[:]),
)


@numba.jitclass(spec)
class BlockList(object):
    """
    BlockList algorithm partitions space into squares and sorts points into
    the square they belong. This allows fast neighbourhood search because we
    only have to search current and neighbouring rectangles for points.
    """

    def __init__(self, points, cell_size):
        assert cell_size > 0
        assert points.ndim == 2
        assert points.shape[1] == 2

        index_list, count, offset, x_min, x_max = block_list(points, cell_size)
        self.cell_width = cell_size
        self.index_list = index_list
        self.count = count
        self.offset = offset
        self.x_min = x_min
        self.x_max = x_max
        self.shape = x_max + 1

    def get_block(self, indices):
        r"""
        Multidimensional indexing

        1-D: [...]
        dims: n0
        key: x0
        index: x0

        2-D: [[...], [...]]
        dims: (n0, n1)
        key: (x0, x1)
        index: x0 * n1 + x1

        3-D: [[[...], [...]], [[...], [...]]]
        dims: (n0, n1, n2)
        key:(x0, x1, x2)
        index: x0 * n1 * n2 + x1 * n2 + x2
              (x0 * n1 + x1) * n2 + x2

        Args:
            indices (numpy.ndarray | tuple):

        Returns:
            numpy.ndarray:
        """
        index = indices[0]
        for j in range(1, len(indices)):
            index *= self.shape[j]
            index += indices[j]
        start = self.offset[index]
        end = start + self.count[index]
        return self.index_list[start:end]


class ConvexHull(object):
    r"""
    Convex hull algorithm

    http://doi.org/10.1016/j.asoc.2009.07.004
    """
    pass
