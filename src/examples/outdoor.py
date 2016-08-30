from shapely.geometry import Polygon

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
