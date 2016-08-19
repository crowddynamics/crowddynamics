from src.multiagent import MultiAgentSimulation, random_unit_vector
from src.structure.area import Rectangle


class Outdoor(MultiAgentSimulation):
    def __init__(self, queue, size, width, height, model, body):
        # TODO: Periodic boundaries
        super().__init__(queue)
        domain = Rectangle((0.0, width), (0.0, height))
        target_direction = random_unit_vector(size)
        kw = {'amount': size,
              'area': domain,
              'target_direction': target_direction}

        self.configure_domain(domain)
        self.configure_goals(None)
        self.configure_obstacles(None)
        self.configure_exits(None)

        self.configure_agent(size, body)
        self.configure_agent_model(model)
        self.configure_agent_positions(kw)

        self.configure_navigation()
        self.configure_orientation()
