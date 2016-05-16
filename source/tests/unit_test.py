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
        from source.field.wall import RoundWall
        rp = np.array(((0.0, 0.0, 1.0),
                       (0.0, 0.0, 1.0)))
        rw = RoundWall(rp)

        # self.assertEqual(rw.deconstruct(0))
        # self.assertEqual(rw.deconstruct(1))
        with self.assertRaises(IndexError):
            rw.deconstruct(2)

    def test_linear_wall(self):
        from source.field.wall import LinearWall
        lp = np.array((((0.0, 0.0), (1.0, 2.0)),
                       ((0.0, 0.0), (2.0, 0.0))))
        lw = LinearWall(lp)

        # self.assertEqual(lw.deconstruct(0))
        # self.assertEqual(lw.deconstruct(1))
        with self.assertRaises(IndexError):
            lw.deconstruct(2)

    def test_agent(self):
        from source.field.agent import agent_struct
        num = 1.0
        arr1d = np.ones(10, np.float64)
        arr2d = np.ones(20, np.float64).reshape(10, 2)
        for var in (num, arr1d):
            # Should not raise errors
            agent = agent_struct(var, var, arr2d, arr2d, var, arr2d)
            # agent.mass
            # agent.radius
            # agent.position
            # agent.velocity
            # agent.goal_velocity
            # agent.goal_direction
            # agent.size

    def test_forces(self):
        pass


if __name__ == '__main__':
    unittest.main()
