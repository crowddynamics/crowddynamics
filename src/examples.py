import numpy as np
from shapely.geometry import Polygon, LineString, Point

from src.core.game import EgressGame
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

        self.set_field(domain)
        self.set_algorithms()

        self.set_body(size, body)
        self.set_model(model)
        self.set_agents(**kwargs)


class Hallway(MultiAgentSimulation):
    def __init__(self, queue, size, width, height, model, body):
        super().__init__(queue)
        domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])

        obstacles = (
            LineString([(0, 0), (width, 0)]),
            LineString([(0, height), (width, height)]),
        )

        spawn = (
            Polygon([(1.1, 0),
                     (1.1, height),
                     (width // 2 - 1, height),
                     (width // 2 - 1, 0)]),
            Polygon([(width // 2 + 1, 0),
                     (width // 2 + 1, height),
                     (width - 1.1, height),
                     (width - 1.1, 0)]),
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

        self.set_field(domain=domain, obstacles=obstacles)
        self.set_algorithms()

        self.set_body(size, body)
        self.set_model(model)
        for kw in kwargs:
            self.set_agents(**kw)


class Rounding(MultiAgentSimulation):
    def __init__(self, queue, size, width, height, model, body):
        super().__init__(queue)

        domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
        exits = LineString([(0, height / 2), (0, height)])
        obstacles = [
            LineString([(0, height / 2), (width * 3 / 4, height / 2)]),
            domain.exterior - exits
        ]

        spawn = Polygon([(0, 0),
                         (0, height / 2),
                         (width / 2, height / 2),
                         (width / 2, 0)])
        kwargs = {
            'size': size,
            'surface': spawn,
            'target_direction': None,
            'orientation': 0
        }

        self.set_field(domain, obstacles, exits)
        self.set_algorithms(navigation="static")

        self.set_body(size, body)
        self.set_model(model)
        self.set_agents(**kwargs)


class RoomEvacuation(MultiAgentSimulation):
    def __init__(self, queue, size, width, height, model, body, spawn_shape,
                 door_width, exit_hall_width):
        super(RoomEvacuation, self).__init__(queue)

        room = Polygon([(0, 0), (0, height), (width, height), (width, 0), ])
        hall = Polygon([(width, (height - door_width) / 2),
                        (width, (height + door_width) / 2),
                        (width + exit_hall_width, (height + door_width) / 2),
                        (width + exit_hall_width, (height - door_width) / 2), ])

        door = LineString([
            (width + exit_hall_width, (height - door_width) / 2),
            (width + exit_hall_width, (height + door_width) / 2),
        ])

        domain = room | hall
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

        self.set_field(domain=domain, obstacles=obstacles, exits=door)
        self.set_algorithms(navigation="static")

        self.set_body(size, body)
        self.set_model(model)
        self.set_agents(**kwargs)


class RoomEvacuationGame(RoomEvacuation):
    def __init__(self, queue, size, width, height, model, body, spawn_shape,
                 door_width, exit_hall_width, t_aset_0, interval,
                 neighbor_radius, neighborhood_size):
        super(RoomEvacuationGame, self).__init__(
            queue, size, width, height, model, body, spawn_shape, door_width,
            exit_hall_width)
        # FIXME: Exit door
        door = np.array(([(width, (height - door_width) / 2),
                          (width, (height + door_width) / 2), ]))
        self.game = EgressGame(self, door, t_aset_0, interval, neighbor_radius,
                               neighborhood_size)
