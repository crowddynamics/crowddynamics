from src.initialize import initialize_agent, random_unit_vector
from src.simulation import Simulation
from src.structure.area import Rectangle


def outdoor(size, width, height, agent_model, body_type, path="", name="outdoor",
            **kwargs):
    domain = Rectangle((0.0, width), (0.0, height))
    target_direction = random_unit_vector(size)

    populate_kwargs_list = {'amount': size,
                            'area': domain,
                            'target_direction': target_direction}
    agent = initialize_agent(size, populate_kwargs_list, agent_model=agent_model,
                             body_type=body_type)

    return Simulation(agent, name=name, dirpath=path, domain=domain, **kwargs)
