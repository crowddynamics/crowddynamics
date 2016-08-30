import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Polygon

from src.multiagent.simulation import MultiAgentSimulation


class Hallway(MultiAgentSimulation):
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
             'orientation': 0},
            {'size': size // 2,
             'surface': spawn[1],
             'target_direction': np.array((-1.0, 0.0)),
             'orientation': np.pi},
        )

        self.set_domain(domain)
        self.set_goals(goals)
        self.set_obstacles(obstacles)
        self.set_body(size, body)
        self.set_model(model)
        for kw in kwargs:
            self.set(**kw)

        self.set_navigation()
        self.set_orientation()
