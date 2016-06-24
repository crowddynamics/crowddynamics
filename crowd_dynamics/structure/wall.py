import numpy as np
from numba import float64, int64
from numba import jitclass

from ..core.vector2d import rotate90, dot2d, angle

"""
Static obstacles
"""

spec_round = [
    ("params", float64[:, :]),
    ("cols", int64),
    ("rows", int64),
    ("size", int64),
    ("wall", float64[:, :]),
]

# Same for round and linear
wall_attr_names = [item[0] for item in spec_round]


@jitclass(spec_round)
class RoundWall(object):
    def __init__(self, round_params):
        self.params = round_params
        self.cols = 3
        self.rows = len(self.params)
        self.wall = np.zeros((self.rows, self.cols))
        self.size = self.rows
        self.construct()

    def construct(self):
        for i in range(self.size):
            self.wall[i] = self.params[i]

    def deconstruct(self, index):
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds. "
                             "Index should be: 0 <= index < size.")
        w = self.wall[index]
        p, r = w[0:2], w[2]
        return p, r

    def distance(self, i, x):
        p, r = self.deconstruct(i)
        q = x - p
        d_iw = np.hypot(q[0], q[1])
        d_iw -= r
        return d_iw

    def distance_with_normal(self, i, x):
        p, r = self.deconstruct(i)
        q = x - p
        d_iw = np.hypot(q[0], q[1])
        n_iw = q / d_iw
        d_iw -= r
        return d_iw, n_iw

    def relative_position(self, i, x, v):
        p, r = self.deconstruct(i)
        return p - x, r


spec_linear = (
    ("params", float64[:, :, :]),
    ("cols", int64),
    ("rows", int64),
    ("size", int64),
    ("wall", float64[:, :]),
)


@jitclass(spec_linear)
class LinearWall(object):
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

    def relative_position(self, i, x, v):
        p_0, p_1, t_w, n_w, l_w = self.deconstruct(i)
        r = 0.0

        # v is zero vector
        if np.all(v == 0.0):
            return np.zeros(2), r

        q_0 = p_0 - x
        q_1 = p_1 - x

        n_iw = -np.sign(dot2d(n_w, q_0)) * n_w

        angles = np.array((angle(q_0), angle(q_1), angle(n_iw)))

        if angles[0] == angles[1]:
            if np.hypot(q_0[0], q_0[1]) < np.hypot(q_1[0], q_1[1]):
                return q_0, r
            else:
                return q_1, r

        angles -= angle(v)  # Angle of velocity vector
        angles %= 2 * np.pi

        args = (np.argmin(angles), np.argmax(angles))

        if 2 in args:
            if 0 in args:
                return q_0, r
            else:
                return q_1, r
        else:
            arr = np.zeros((2, 2))
            arr[:, 0] = v
            arr[:, 1] = p_0 - p_1
            # TODO: Is matrix invertable?
            arr = np.linalg.inv(arr)
            a, _ = np.dot(arr, q_0)
            return a * v, r
