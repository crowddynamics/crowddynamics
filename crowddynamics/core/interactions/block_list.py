r"""
Block List
----------
Since crowd simulations are only dependent on interactions with agents close by
we can partition the space into smaller chunk in order to avoid having to loop
with agents far a away.

.. tikz:: Example of block list partitioning

   % Measures
   \draw[<->] (-0.1, 0) -- node[left] {$n$} ++(0, 8);
   \draw[<->] (0, -0.1) -- node[below] {$m$} ++(12, 0);
   \draw[<->] (-0.3, 0) -- node[left] {$c$} ++(0, 1);
   \draw[<->] (0, -0.3) -- node[below] {$c$} ++(1, 0);
   % Grid
   \draw[color=gray!20] (0, 0) grid (12, 8);
   \draw[thick, <->] (13, 6) 
                     -- node[left] {$y$} ++(0, -1) 
                     -- node[below] {$x$} ++(1, 0);
   \draw[] (4.5, 5.7) circle (1pt) node[below] {$\mathbf{p}_i$};
   \draw[dashed] (4.5, 5.7) circle (1);
   \draw[<->] (4.5, 5.7) -- node[below] {$r$} ++(45:1);

Block list means partitioning scheme where the bounding box of points
:math:`\mathbf{p}_i` is partitioned into a grid of ``shape`` :math:`n \times m` 
with cell size of :math:`c`.

Algorithm

0. Cell size is equal to the interaction range :math:`c = r`.
1. Get the indices of the block the point :math:`\mathbf{p}_i` belongs to 
   
   .. math::
      \mathbf{l}_i = \left\lfloor \frac{\mathbf{p}_i}{c} \right\rfloor, \quad 
      i \in [0, N-1]

2. Crate array ``count`` of counts of how many points belong to each block.
3. Crate array ``index_list`` for indices of agents in each block.

   .. tikz:: 
      \node[above] () at (0.5, 0) {$\dots$};
      \node[above] () at (5.5, 0) {$\dots$};
      \draw[<->] (1, -0.1) -- node[below] {count} ++(4, 0);
      \draw[] (1, 0) -- ++(0, -0.5) node[below] {offset};
      \draw[color=gray!20] (0, 0) grid ++(6, 1);
      \node[above] () at (1.5, 0) {$i_0$};
      \node[above] () at (2.5, 0) {$i_1$};
      \node[above] () at (3.5, 0) {$i_2$};
      \node[above] () at (4.5, 0) {$i_3$};

4. Crate array ``offset`` from cumulative sum of counts to track the starting
   index of indices in ``index_list`` array when querying agents in each block.


Iteration
^^^^^^^^^
Iterating over block list.

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
    # TODO: https://docs.python.org/3/library/collections.abc.html#module-collections.abc

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
