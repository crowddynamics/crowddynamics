from functools import partial

import numba
import numpy as np
from shapely.geometry import Polygon, LineString, Point

from src.core.game import EgressGame
from src.core.vector2D import length, normalize, rotate90

from src.multiagent.simulation import MultiAgentSimulation


class Outdoor(MultiAgentSimulation):
    def __init__(self, queue, size, width, height, model, body):
        # TODO: Periodic boundaries
        super().__init__(queue)

        domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
        kwargs = {
            "size": size,
            "surface": domain,
            "target_direction": "random",
            "velocity": "auto",
        }

        self.set_domain(domain)
        self.set_body(size, body)
        self.set_model(model)
        self.set(**kwargs)

        self.set_navigation()
        self.set_orientation()


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

        self.set_obstacles_to_linear_walls()


class RoomEvacuation(MultiAgentSimulation):
    def __init__(self, queue, size, width, height, model, body, spawn_shape,
                 door_width, exit_hall_width):
        super(RoomEvacuation, self).__init__(queue)

        room = Polygon([(0, 0), (0, height), (width, height), (width, 0), ])
        hall = Polygon([(width, (height - door_width) / 2),
                        (width, (height + door_width) / 2),
                        (width + exit_hall_width, (height + door_width) / 2),
                        (width + exit_hall_width, (height - door_width) / 2), ])

        door = LineString([(width + exit_hall_width, (height - door_width) / 2),
                           (width + exit_hall_width,
                            (height + door_width) / 2), ])

        domain = room | hall
        goals = hall
        obstacles = (room | hall).exterior - door

        spawn = room
        if spawn_shape == "circ":
            spawn = room & Point((width, height / 2)).buffer(height / 2)

        kwargs = {
            'size': size,
            'surface': spawn,
            'target_direction': None,
            'orientation': 0
        }

        self.set_domain(domain)
        self.set_goals(goals)
        self.set_obstacles(obstacles)
        self.set_exits(door)
        self.set_body(size, body)
        self.set_model(model)
        self.set(**kwargs)

        self.set_navigation("static")
        self.set_orientation()

        self.set_obstacles_to_linear_walls()


class RoomEvacuationGame(RoomEvacuation):
    def __init__(self, queue, size, width, height, model, body, spawn_shape,
                 door_width, exit_hall_width, t_aset_0, interval,
                 neighbor_radius, neighborhood_size):
        super(RoomEvacuationGame, self).__init__(
            queue, size, width, height, model, body, spawn_shape, door_width,
            exit_hall_width)
        # FIXME: Exit door
        door = LineString([(width, (height - door_width) / 2),
                           (width, (height + door_width) / 2), ]),
        self.game = EgressGame(self.agent, door, t_aset_0, interval,
                               neighbor_radius, neighborhood_size)
