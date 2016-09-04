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


@numba.jit(nopython=True)
def _direction_update(agent, target, mid, r_mid, c_rect, r_rect):
    target_direction = np.zeros(agent.shape)
    for i in range(agent.size):
        x = agent.position[i]
        cond1 = (x - c_rect) <= r_rect
        cond2 = length(mid - x) > r_mid
        if np.all(cond1) and cond2:
            # if inside walls and not inside circle near exit
            target_direction[i] = normalize(target - x)
        else:
            target_direction[i] = np.array((1.0, 0.0))
    return target_direction


class Navigation:
    def __init__(self, agent, door, door_width, width, height):
        self.agent = agent
        # Navigation algorithm
        door1 = np.array(door[1], dtype=np.float64)
        door0 = np.array(door[0], dtype=np.float64)
        unit = normalize(door1 - door0)
        normal = rotate90(unit)
        mid = (door0 + door1) / 2  # Mid point of the two doors
        r_max = 0.27  # Max agent radius
        target = mid + r_max * normal
        c_rect = r_rect = np.array((width / 2, height / 2))
        r_mid = door_width / 2

        self.func = partial(_direction_update, target=target, mid=mid,
                            r_mid=r_mid, c_rect=c_rect, r_rect=r_rect)

    def update(self):
        self.agent.target_direction = self.func(self.agent)


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

        # target = np.asarray(exit_door)
        # navigation = Navigation(self.agent,
        #                         target,
        #                         door_width,
        #                         width,
        #                         height)
        # self.set_navigation(navigation)

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
