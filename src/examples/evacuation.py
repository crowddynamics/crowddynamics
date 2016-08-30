from functools import partial

import numba
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from src.core.game import EgressGame
from src.core.vector2d import rotate90, normalize, length
from src.multiagent.simulation import MultiAgentSimulation


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

        door = (
            LineString([(width, (height - door_width) / 2),
                        (width, (height + door_width) / 2), ]),
            LineString([(width + exit_hall_width, (height - door_width) / 2),
                        (width + exit_hall_width, (height + door_width) / 2), ])
        )

        domain = room | hall
        goals = hall
        exit_door = door[0]
        obstacles = (room | hall).exterior - door[1]

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
        self.set_exits(exit_door)
        self.set_body(size, body)
        self.set_model(model)
        self.set(**kwargs)

        # FIXME
        # navigation = Navigation(self.agent, door, door_width, width, height)
        # self.set_navigation(navigation)
        # self.set_orientation()


class RoomEvacuationGame(RoomEvacuation):
    def __init__(self, queue, size, width, height, model, body, spawn_shape,
                 door_width, exit_hall_width, t_aset_0, interval,
                 neighbor_radius, neighborhood_size):
        super(RoomEvacuationGame, self).__init__(
            queue, size, width, height, model, body, spawn_shape, door_width,
            exit_hall_width)
        self.game = EgressGame(self.agent, self.exits[0], t_aset_0, interval,
                               neighbor_radius, neighborhood_size)

        # def configure_saving(self, dirpath):
        #     super(RoomEvacuationGame, self).configure_saving(dirpath)
        #     attrs_egress = Attrs(self.game.attrs, Intervals(1.0))
        #     recordable = ("strategy", "t_evac")
        #     for attr in recordable:
        #         attrs_egress[attr] = Attr(attr, True, True)
        #     self.hdfstore.save(self.game, attrs_egress)
