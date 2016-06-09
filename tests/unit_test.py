import unittest

from src.params import Params
from src.struct.agent import Agent, agent_attr_names
from src.struct.constant import Constant, constant_attr_names
from src.struct.wall import LinearWall, RoundWall, wall_attr_names


class MyTestCase(unittest.TestCase):
    params = Params(100, 100)

    def test_attributes(self):
        constant = Constant()
        agent = Agent(*self.params.agent(100))
        round_wall = RoundWall(self.params.round_wall(5, 0.1, 0.3))
        linear_wall = LinearWall(self.params.linear_wall(5))

        for name in constant_attr_names:
            self.assertTrue(hasattr(constant, name))

        for name in agent_attr_names:
            self.assertTrue(hasattr(agent, name))

        for name in wall_attr_names:
            self.assertTrue(hasattr(linear_wall, name))
            self.assertTrue(hasattr(round_wall, name))


if __name__ == '__main__':
    unittest.main()
