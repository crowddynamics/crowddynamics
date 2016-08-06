import numpy as np

from crowd_dynamics.simulation import Simulation
from crowd_dynamics.structure.area import Rectangle
from crowd_dynamics.structure.initialize import initialize_agent
from crowd_dynamics.structure.wall import LinearWall


def hallway(size,
            width,
            height,
            agent_model="circular",
            body_type="adult",
            path="",
            name="hallway",
            **kwargs):
    domain = Rectangle((0, width), (0, height))
    linear_params = np.array((
        ((0.0, 0.0), (width, 0.0)),
        ((0.0, height), (width, height)),
    ))
    walls = LinearWall(linear_params)

    # Goal
    goals = (
        Rectangle((0, 1), (0, height)),
        Rectangle((width, width + 1), (0, height))
    )

    spawn1 = Rectangle((1.0, width // 2), (0.0, height))
    spawn2 = Rectangle((width // 2, width - 1.0), (0.0, height))
    populate_kwargs_list = (
        {'amount': size // 2,
         'area': spawn1,
         'target_direction': np.array((1.0, 0.0)),
         'body_angle': 0},
        {'amount': size // 2,
         'area': spawn2,
         'target_direction': np.array((-1.0, 0.0)),
         'body_angle': np.pi},
    )
    agent = initialize_agent(size, populate_kwargs_list, body_type=body_type,
                             agent_model=agent_model, walls=walls)

    return Simulation(agent, wall=walls, goals=goals, name=name, dirpath=path,
                      domain=domain, **kwargs)
