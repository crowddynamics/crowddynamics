import unittest

import numpy as np


class MyTestCase(unittest.TestCase):
    def test_constants(self):
        from source.parameters import Constant
        c = Constant()
        self.assertIsInstance(c.tau_adj, float)
        self.assertIsInstance(c.k, float)
        self.assertIsInstance(c.tau_0, float)
        self.assertIsInstance(c.mu, float)
        self.assertIsInstance(c.kappa, float)
        self.assertIsInstance(c.a, float)
        self.assertIsInstance(c.b, float)

    def test_round_wall(self):
        from source.struct.wall import RoundWall
        rp = np.array(((0.0, 0.0, 1.0),
                       (0.0, 0.0, 1.0)))
        round_wall = RoundWall(rp)

        # self.assertEqual(rw.deconstruct(0))
        # self.assertEqual(rw.deconstruct(1))
        with self.assertRaises(IndexError):
            round_wall.deconstruct(2)

    def test_linear_wall(self):
        from source.struct.wall import LinearWall
        lp = np.array((((0.0, 0.0), (1.0, 2.0)),
                       ((0.0, 0.0), (2.0, 0.0))))
        linear_wall = LinearWall(lp)

        # self.assertEqual(lw.deconstruct(0))
        # self.assertEqual(lw.deconstruct(1))
        with self.assertRaises(IndexError):
            linear_wall.deconstruct(2)

    def test_agent(self):
        from source.struct.agent import agent_struct, initial_position
        from source.struct.wall import LinearWall

        amount = 10
        x_dims = (0, 100)
        y_dims = (0, 100)
        radii = (1, 0.5 * np.linspace(1, 11, amount))

        lp = np.array((((0.0, 0.0), (100.0, 0.0)),
                       ((0.0, 0.0), (0, 100.0))))
        lw = LinearWall(lp)

        num = 1.0
        arr1d = np.ones(amount, np.float64)
        arr2d = np.ones(2 * amount, np.float64).reshape(10, 2)

        for var in (num, arr1d):
            # Test agent struct initialization
            agent = agent_struct(var, var, arr2d, arr2d, var, arr2d)

        for radius in radii:
            # Without walls
            position = initial_position(amount, x_dims, y_dims, radius)

        for radius in radii:
            # With walls
            position = initial_position(amount, x_dims, y_dims, radius, lw)

    def test_forces(self):
        pass


if __name__ == '__main__':
    unittest.main()
