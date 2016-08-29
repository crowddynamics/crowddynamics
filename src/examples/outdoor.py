from shapely.geometry import Polygon

from src.multiagent.configure import ConfigAgent, ConfigField
from src.multiagent.simulation import MultiAgentSimulation, random_unit_vector
from src.multiagent.surface import Rectangle


class Outdoor(MultiAgentSimulation):
    def __init__(self, queue, size, width, height, model, body):
        # TODO: Periodic boundaries
        super().__init__(queue)

        domain = Rectangle((0.0, width), (0.0, height))
        target_direction = random_unit_vector(size)
        positions = {'amount': size,
                     'area': domain,
                     'target_direction': target_direction}

        self.configure_domain(domain)
        self.configure_goals(None)
        self.configure_obstacles(None)
        self.configure_exits(None)

        self.configure_agent(size, body)
        self.configure_agent_model(model)
        self.configure_agent_positions(positions)

        self.configure_navigation()
        self.configure_orientation()


class Outdoor2(MultiAgentSimulation):
    def __init__(self, queue, size, width, height, model, body):
        # TODO: Periodic boundaries
        super().__init__(queue)

        domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])

        field = ConfigField()
        field.set_domain(domain)

        agent = ConfigAgent(field, size, model, body)
        kwargs = {
            "size": size,
            "surface": domain,
            "target_direction": "random",
            "velocity": "auto",
        }
        agent.set(**kwargs)

        # self.configure_navigation()
        # self.configure_orientation()
