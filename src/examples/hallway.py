import numpy as np

from src.multiagent import MultiAgentSimulation
from src.structure.area import Rectangle
from src.structure.obstacle import LinearObstacle


class Hallway(MultiAgentSimulation):
    def __init__(self, size, width, height, model, body):
        super().__init__()
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
