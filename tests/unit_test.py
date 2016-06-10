import unittest

from src.parameters import Parameters
from src.struct.agent import Agent, agent_attr_names
from src.struct.constant import Constant, constant_attr_names
from src.struct.wall import LinearWall, RoundWall, wall_attr_names


class MyTestCase(unittest.TestCase):
    def test_attributes(self):
        params = Parameters(100, 100)
        constant = Constant()
        agent = Agent(*params.agent(100))
        round_wall = RoundWall(params.round_wall(5, 0.1, 0.3))
        linear_wall = LinearWall(params.linear_wall(5))

        for name in constant_attr_names:
            self.assertTrue(hasattr(constant, name))

        for name in agent_attr_names:
            self.assertTrue(hasattr(agent, name))

        for name in wall_attr_names:
            self.assertTrue(hasattr(linear_wall, name))
            self.assertTrue(hasattr(round_wall, name))


if __name__ == '__main__':
    unittest.main()
