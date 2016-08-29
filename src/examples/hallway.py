import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Polygon

from src.multiagent.configure import ConfigField, ConfigAgent
from src.multiagent.curve import LinearObstacle
from src.multiagent.simulation import MultiAgentSimulation
from src.multiagent.surface import Rectangle


class Hallway(MultiAgentSimulation):
    def __init__(self, queue, size, width, height, model, body):
        super().__init__(queue)

        domain = Rectangle((0, width), (0, height))

        goals = (
            Rectangle((0, 1), (0, height)),
            Rectangle((width - 1, width), (0, height))
        )

        linear_params = np.array((
            ((0.0, 0.0), (width, 0.0)),
            ((0.0, height), (width, height)),
        ))
        walls = LinearObstacle(linear_params)

        spawn1 = Rectangle((1.1, width // 2 - 1), (0.0, height))
        spawn2 = Rectangle((width // 2 + 1, width - 1.1), (0.0, height))
        kw = (
            {'amount': size // 2,
             'area': spawn1,
             'target_direction': np.array((1.0, 0.0)),
             'body_angle': 0},
            {'amount': size // 2,
             'area': spawn2,
             'target_direction': np.array((-1.0, 0.0)),
             'body_angle': np.pi},
        )

        self.configure_domain(domain)
        self.configure_goals(goals)
        self.configure_obstacles(walls)
        self.configure_exits()

        self.configure_agent(size, body)
        self.configure_agent_model(model)
        self.configure_agent_positions(kw)

        self.configure_navigation()
        self.configure_orientation()


class Hallway2(MultiAgentSimulation):
    def __init__(self, queue, size, width, height, model, body):
        super().__init__(queue)

        domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])

        goals = (
            Polygon([(0, 0), (0, height), (width, height), (1, width)]),
            Polygon([(0, 0), (0, height), (1, height), (1, 0)]),
        )

        obstacles = (
            LineString([(0, 0), (width, 0)]),
            LineString([(0, height), (width, height)]),
        )

        field = ConfigField()
        field.set_domain(domain)
        field.set_goals(goals)
        field.set_obstacles(obstacles)

        agent = ConfigAgent(field, size, model, body)
        spawn = (
            Polygon([(1.1, 0), (1.1, height),
                     (width // 2 - 1, height), (width // 2 - 1, 0)]),
            Polygon([(width // 2 + 1, 0), (width // 2 + 1, height),
                     (width - 1.1, height), (width - 1.1, 0)]),
        )
        kwargs = (
            {'size': size // 2,
             'surface': spawn[0],
             'target_direction': np.array((1.0, 0.0)),
             'body_angle': 0},
            {'size': size // 2,
             'surface': spawn[1],
             'target_direction': np.array((-1.0, 0.0)),
             'body_angle': np.pi},
        )
        for kw in kwargs:
            agent.set(**kw)

        # self.configure_navigation()
        # self.configure_orientation()
