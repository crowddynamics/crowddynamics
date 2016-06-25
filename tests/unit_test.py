import unittest

from crowd_dynamics.parameters import Parameters
from crowd_dynamics.structure.agent import Agent, agent_attr_names
from crowd_dynamics.structure.wall import LinearWall, RoundWall, wall_attr_names


class MyTestCase(unittest.TestCase):
    def test_attributes(self):
        parameters = Parameters(100, 100)
        agent = Agent(*parameters.agent(100))
        round_wall = RoundWall(parameters.random_round_wall(5, 0.1, 0.3))
        linear_wall = LinearWall(parameters.random_linear_wall(5))

        for name in agent_attr_names:
            self.assertTrue(hasattr(agent, name))

        for name in wall_attr_names:
            self.assertTrue(hasattr(linear_wall, name))
            self.assertTrue(hasattr(round_wall, name))


if __name__ == '__main__':
    unittest.main()
