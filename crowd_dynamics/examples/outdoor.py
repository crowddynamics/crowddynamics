from crowd_dynamics.simulation import Simulation
from crowd_dynamics.structure.area import Rectangle
from crowd_dynamics.structure.initialize import initialize_agent, random_unit_vector


def initialize(size, width, height, model="circular", body_type="adult",
               path="", name="outdoor", **kwargs):
    domain = Rectangle((0.0, width), (0.0, height))
    target_direction = random_unit_vector(size)

    populate_kwargs_list = {'amount': size, 'area': domain,
                            'target_direction': target_direction}
    agent = initialize_agent(size, populate_kwargs_list, model=model,
                             body_type=body_type)

    return Simulation(agent, name=name, dirpath=path, domain=domain, **kwargs)
