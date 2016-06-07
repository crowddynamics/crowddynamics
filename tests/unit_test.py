import unittest

from src.params import Params
from src.struct.agent import agent_struct
from src.struct.constant import Constant, constant_attr_names
from src.struct.wall import LinearWall
from src.struct.wall import RoundWall


class MyTestCase(unittest.TestCase):
    params = Params(100, 100)

    def test_constants(self):
        constant = Constant()
        for name in constant_attr_names:
            self.assertTrue(hasattr(constant, name))

    def test_round_wall(self):
        round_wall = RoundWall(self.params.round_wall(10))
        # self.assertTrue(True)

    def test_linear_wall(self):
        linear_wall = LinearWall(self.params.linear_wall(10))
        # self.assertTrue(True)

    def test_agent(self):
        agent = agent_struct(*self.params.agent())
        round_wall = RoundWall(self.params.round_wall(5))
        linear_wall = LinearWall(self.params.linear_wall(5))
        # self.assertTrue(True)

    def test_forces(self):
        pass


if __name__ == '__main__':
    unittest.main()
