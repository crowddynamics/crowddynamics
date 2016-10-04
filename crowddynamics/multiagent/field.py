import numpy as np
from numba import float64, int64
from numba import jitclass

from crowddynamics.core.vector2D import rotate90, dot2d

spec_linear = (
    ("params", float64[:, :, :]),
    ("cols", int64),
    ("rows", int64),
    ("size", int64),
    ("wall", float64[:, :]),
)


@jitclass(spec_linear)
class LineObstacle(object):
    def __init__(self, linear_params):
        self.params = linear_params
        self.cols = 9
        self.rows = len(self.params)
        self.wall = np.zeros(shape=(self.rows, self.cols))
        self.size = self.rows
        self.construct()

    def construct(self):
        for i in range(self.size):
            p = self.params[i]
            d = p[1] - p[0]  # Vector from p_0 to p_1
            l_w = np.hypot(d[1], d[0])  # Length of the wall
            t_w = d / l_w  # Tangential unit-vector
            n_w = rotate90(t_w)  # Normal unit-vector
            w = self.wall[i]  # Set values to wall array
            w[0:2], w[2:4], w[4:6], w[6:8], w[8] = p[0], p[1], t_w, n_w, l_w

    def deconstruct(self, index):
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds. "
                             "Index should be: 0 <= index < size.")
        w = self.wall[index]
        p_0, p_1, t_w, n_w, l_w = w[0:2], w[2:4], w[4:6], w[6:8], w[8]
        return p_0, p_1, t_w, n_w, l_w

    def distance(self, i, x):
        """
        Linear wall i's distance from Cartesian coordinate x.
        """
        p_0, p_1, t_w, n_w, l_w = self.deconstruct(i)

        q_0 = x - p_0
        q_1 = x - p_1

        l_t = - dot2d(t_w, q_1) - dot2d(t_w, q_0)

        if l_t > l_w:
            d_iw = np.hypot(q_0[0], q_0[1])
        elif l_t < -l_w:
            d_iw = np.hypot(q_1[0], q_1[1])
        else:
            l_n = dot2d(n_w, q_0)
            d_iw = np.abs(l_n)

        return d_iw

    def distance_with_normal(self, i, x):
        p_0, p_1, t_w, n_w, l_w = self.deconstruct(i)

        q_0 = x - p_0
        q_1 = x - p_1

        l_t = - dot2d(t_w, q_1) - dot2d(t_w, q_0)

        if l_t > l_w:
            d_iw = np.hypot(q_0[0], q_0[1])
            n_iw = q_0 / d_iw
        elif l_t < -l_w:
            d_iw = np.hypot(q_1[0], q_1[1])
            n_iw = q_1 / d_iw
        else:
            l_n = dot2d(n_w, q_0)
            d_iw = np.abs(l_n)
            n_iw = np.sign(l_n) * n_w

        return d_iw, n_iw
