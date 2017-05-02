"""Spatial partitioning algorithms.

Since crowd simulations are only dependent on interactions with agents close by
we can partition the space into smaller chunk in order to avoid having to loop
with agents far a away.
    
Todo: 
    - Convex Hull algorithm: http://doi.org/10.1016/j.asoc.2009.07.004
    - https://docs.python.org/3/library/collections.abc.html#module-collections.abc
"""
from collections import defaultdict, MutableSequence
from itertools import product

import numba
import numpy as np
from numba import f8, i8
from numba.types import Tuple


class MutableBlockList(object):
    """Mutable blocklist (or spatial grid hash) implementation.

    Dictionary where the key is index of a block/cell and values are a list of
    items belonging to that block/cell.

    >>> {(0, 1): [1, 3, 4], (1, 2): [2]}

    """

    def __init__(self, cell_size, default_list=list):
        """Initialize

        Args:
            cell_size (float): 
            default_list (Callable[MutableSequence]): 
                Must have append method. For example
                - ``list``
                - ``SortedList``
                - ``lambda: array(typecode)``
        """
        assert cell_size > 0
        assert callable(default_list)

        self._cell_size = cell_size
        self._list = default_list
        self._blocks = defaultdict(default_list)

        self._str = \
            "cell_size: {cell_size}\n" \
            "default_list: {default_list}".format(
                cell_size=cell_size, default_list=default_list)

    @staticmethod
    def _transform(value, cell_size):
        """Key transform function

        Args:
            value: Iterable of numbers 
            cell_size (float): 

        Returns:
            tuple: 
        """
        try:
            return tuple(elem // cell_size for elem in value)
        except:
            raise KeyError

    @staticmethod
    def _nearest_blocks(index, radius):
        """Keys of nearest blocks

        Args:
            index (tuple): 
            radius (int):

        Yields:
            tuple:
        """
        ranges = (range(-radius, radius + 1) for _ in range(len(index)))
        for i in product(*ranges):
            yield tuple(map(sum, zip(index, i)))

    def __setitem__(self, key, value):
        """Add value to blocklist"""
        index = self._transform(key, self._cell_size)
        self._blocks[index].append(value)

    def __getitem__(self, item):
        """Get value in the same block as item"""
        index = self._transform(item, self._cell_size)
        return self._blocks[index]

    def nearest(self, item, radius=1):
        """Get values of neighbouring blocks

        Args:
            item: 
            radius (int): 

        Returns:
            MutableSequence: 
        """
        index = self._transform(item, self._cell_size)
        return sum((self._blocks[key] for key in
                    self._nearest_blocks(index, radius)), self._list())

    def __str__(self):
        return self._str


@numba.jit([(f8[:, :], f8)], nopython=True, nogil=True, cache=True)
def block_list(points, cell_size):
    """Block list partitioning algorithm
    
    BlockList algorithm partitions space into squares and sorts points into
    the square they belong. This allows fast neighbourhood search because we
    only have to search current and neighbouring cells for points.

    Algorithm
    
    1. Find the bounds (minimum and maximum indices) of the points
    2. 

    Args:
        points (numpy.ndarray):
            Array of :math:`N` points :math:`(\mathbf{p}_i \in 
            \mathbb{R}^2)_{i=1,...,N}` (``shape=(size, 2)``) to be block listed

        cell_size (float):
            Positive real number :math:`c > 0`. Width and height of the 
            rectangular mesh.

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray):
            - ``index_list``
            - ``count``
            - ``offset``
            - ``shape``

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
            indices[i, j] = np.int64(points[i, j] / cell_size) - x_min[j]

    shape = x_max - x_min + 1

    # Count how many points go into each cell
    size = np.prod(shape)
    count = np.zeros(size, dtype=np.int64)
    for i in range(n):
        index = indices[i, 0]
        for j in range(1, m):
            index *= shape[j]
            index += indices[i, j]
        count[index] += 1

    # Index list
    index_list = np.zeros(n, dtype=np.int64)
    offset = count.cumsum() - 1  # Offset indices
    for i in range(n):
        index = indices[i, 0]
        for j in range(1, m):
            index *= shape[j]
            index += indices[i, j]
        index_list[offset[index]] = i
        offset[index] -= 1

    offset += 1

    return index_list, count, offset, shape


@numba.jit([i8[:](Tuple((i8, i8)), i8[:], i8[:], i8[:], i8[:]),
            i8[:](i8[:], i8[:], i8[:], i8[:], i8[:])],
           nopython=True, nogil=True, cache=True)
def get_block(indices, index_list, count, offset, shape):
    r"""Multidimensional indexing

    ::

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
    # TODO: Handle index out of bound
    index = indices[0]
    for j in range(1, len(indices)):
        index *= shape[j]
        index += indices[j]
    start = offset[index]
    end = start + count[index]
    return index_list[start:end]


def split_blocklist(parts, index_list, count, offset, shape):
    size = len(index_list)
    part_size = int(size / parts)
    splits = np.zeros(parts, dtype=np.int64)

    num = 0
    _sum = 0
    for i, value in enumerate(count):
        _sum += value
        if _sum >= (num + 1) * part_size:
            splits[num] = i
            num += 1

    return splits
