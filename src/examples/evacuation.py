from functools import partial

import numba
import numpy as np

from src.core.game import EgressGame
from src.core.vector2d import rotate90, normalize, length
from src.multiagent import MultiAgentSimulation
from src.structure.area import Rectangle, Circle
from src.structure.obstacle import LinearExit
from src.structure.obstacle import LinearObstacle
from src.io.attributes import Intervals, Attrs, Attr


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
        domain = Rectangle((0.0, width + exit_hall_width), (0.0, height))

        corner = ((0, 0), (0, height), (width, 0), (width, height))

        door = ((width, (height - door_width) / 2),
                (width, (height + door_width) / 2))

        hall = ((width + exit_hall_width, (height - door_width) / 2),
                (width + exit_hall_width, (height + door_width) / 2))

        linear_params = np.array(
            ((corner[0], corner[1]),
             (corner[0], corner[2]),
             (corner[1], corner[3]),
             (corner[2], door[0]),
             (door[1], corner[3]),
             (door[0], hall[0]),
             (door[1], hall[1]),
             (door[0], door[1]),  # Close the door
             ), dtype=np.float64
        )

        walls = LinearObstacle(linear_params)

        goals = Rectangle((width, width + 2),
                          ((height - door_width) / 2,
                           (height + door_width) / 2))

        # Agents
        spawn = None
        if spawn_shape == "circ":
            spawn = Circle(phi=(np.pi / 2, np.pi / 2 + np.pi),
                           radius=(0, height / 2),
                           center=(width, height / 2))
        elif spawn_shape == "rect":
            spawn = Rectangle(x=(0.0, width),
                              y=(0.0, height))
        else:
            ValueError("Spawn shape not valid.")

        kw = {
            'amount': size,
            'area': spawn,
            'target_direction': None,
            'body_angle': 0
        }

        exit_door = LinearExit(door[0], door[1], 0.27)

        self.configure_domain(domain)
        self.configure_goals(goals)
        self.configure_obstacles(walls)
        self.configure_exits(exit_door)

        self.configure_agent(size, body)
        self.configure_agent_model(model)
        self.configure_agent_positions(kw)

        navigation = Navigation(self.agent, door, door_width, width, height)
        self.configure_navigation(navigation)
        self.configure_orientation()


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
