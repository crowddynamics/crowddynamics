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

"""
from collections import defaultdict, MutableSequence
from itertools import product


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
