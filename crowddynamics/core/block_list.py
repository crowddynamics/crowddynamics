import numpy as np
import numba
from numba import f8, i8


# Multidimensional indexing
# 1-D: [...]
# dims: n0
# key: x0
# index: x0
#
# 2-D: [[...], [...]]
# dims: (n0, n1)
# key: (x0, x1)
# index: x0 * n1 + x1
#
# 3-D: [[[...], [...]], [[...], [...]]]
# dims: (n0, n1, n2)
# key:(x0, x1, x2)
# index: x0 * n1 * n2 + x1 * n2 + x2
#       (x0 * n1 + x1) * n2 + x2

@numba.jit(nopython=True)
def block_list(points, cell_width):
    assert points.ndim == 2
    n, m = points.shape

    x_min = (points[0, :] / cell_width).astype(np.int64)
    x_max = (points[0, :] / cell_width).astype(np.int64)
    for i in range(1, n):
        for j in range(m):
            x = np.int64(points[i, j] / cell_width)
            if x < x_min[j]:
                x_min[j] = x
            if x > x_max[j]:
                x_max[j] = x

    # Blocks
    indices = np.zeros(shape=points.shape, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            x = np.int64(points[i, j] / cell_width) - x_min[j]
            indices[i, j] = x

    x_max = x_max - x_min

    # Count how many points go into each point
    size = np.prod(x_max+1)
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
)


@numba.jitclass(spec)
class BlockList(object):
    def __init__(self, points, cell_width):
        index_list, count, offset, x_min, x_max = block_list(points, cell_width)
        self.cell_width = cell_width
        self.index_list = index_list
        self.count = count
        self.offset = offset
        self.x_min = x_min
        self.x_max = x_max

    def get_block(self, indices):
        index = indices[0]
        for j in range(1, len(indices)):
            index *= self.x_max[j] + 1
            index += indices[j]
        start = self.offset[index]
        end = start + self.count[index]
        return self.index_list[start:end]


def _test():
    n = 10 ** 4
    points = np.random.uniform(0, 100, size=(n, 2))
    cell_width = 1
    bl = BlockList(points, cell_width)
