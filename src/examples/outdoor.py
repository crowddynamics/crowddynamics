from src.simulation.multiagent import MultiAgentSimulation, random_unit_vector
from src.structure.area import Rectangle


class Outdoor(MultiAgentSimulation):
    def __init__(self, size, width, height, model, body):
        super().__init__()
        domain = Rectangle((0.0, width), (0.0, height))
        target_direction = random_unit_vector(size)
        kw = {'amount': size,
              'area': domain,
              'target_direction': target_direction}

        self.configure_domain(domain)
        self.configure_agent(size, body)
        self.configure_agent_model(model)
        self.configure_agent_positions(kw)
