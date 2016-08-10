from functools import partial

import numba
import numpy as np

from src.core.vector2d import rotate90, normalize, length
from src.initialize import initialize_agent
from src.simulation import MultiAgentSimulation
from src.structure.area import Rectangle, Circle
from src.structure.obstacle import LinearWall


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


class RoomEvacuation(MultiAgentSimulation):
    def __init__(self, size, width, height, model, body,
                 spawn_shape="circ", door_width=1.2, exit_hall_width=2):
        super().__init__()
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
             ), dtype=np.float64
        )
        walls = LinearWall(linear_params)

        goals = Rectangle((width, width + 2),
                          ((height - door_width) / 2,
                           (height + door_width) / 2))

        # Agents
        spawn = None
        if spawn_shape == "circ":
            spawn = Circle((np.pi / 2, np.pi / 2 + np.pi), (0, height / 2),
                           (width, height / 2))
        elif spawn_shape == "rect":
            spawn = Rectangle((0.0, width), (0.0, height))
        else:
            ValueError("Spawn shape not valid.")

        kw = {
            'amount': size,
            'area': spawn,
            'target_direction': None,
            'body_angle': 0
        }

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

        self.direction_update = partial(_direction_update, target=target,
                                        mid=mid,
                                        r_mid=r_mid, c_rect=c_rect,
                                        r_rect=r_rect)

        self.configure_domain(domain)
        self.configure_goals(goals)
        self.configure_agent(size, body)
        self.configure_agent_model(model)
        self.configure_obstacles(walls)
        self.configure_agent_positions(kw)


def evacuation(size, width, height, agent_model, body_type,
               spawn_shape="circ", door_width=1.2, exit_hall_width=2,
               egress_model=False, t_aset=60):
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
         ), dtype=np.float64
    )
    walls = LinearWall(linear_params)

    goals = Rectangle((width, width + 2),
                      ((height - door_width) / 2, (height + door_width) / 2))

    # Agents
    spawn = None
    if spawn_shape == "circ":
        spawn = Circle((np.pi / 2, np.pi / 2 + np.pi), (0, height / 2),
                       (width, height / 2))
    elif spawn_shape == "rect":
        spawn = Rectangle((0.0, width), (0.0, height))
    else:
        ValueError("Spawn shape not valid.")

    populate_kwargs_list = {
        'amount': size,
        'area': spawn,
        'target_direction': None,
        'body_angle': 0
    }
    agent = initialize_agent(size, populate_kwargs_list, body_type=body_type,
                             agent_model=agent_model, walls=walls)

    # Navigation algorithm
    door1 = np.array(door[1], dtype=np.float64)
    door0 = np.array(door[0], dtype=np.float64)
    unit = normalize(door1 - door0)
    normal = rotate90(unit)
    mid = (door0 + door1) / 2  # Mid point of the two doors
    r_max = np.max(agent.radius)
    target = mid + r_max * normal
    c_rect = r_rect = np.array((width / 2, height / 2))
    r_mid = door_width / 2

    direction_update = partial(_direction_update, target=target, mid=mid,
                               r_mid=r_mid, c_rect=c_rect, r_rect=r_rect)

    if egress_model:
        from src.structure.obstacle import ExitDoor
        from src.core.egress import EgressGame
        exit_door = ExitDoor(door[0], door[1], np.mean(agent.radius))
        egress_model = EgressGame(agent, exit_door, t_aset, 0.1)
    else:
        egress_model = None

    return MultiAgentSimulation()
