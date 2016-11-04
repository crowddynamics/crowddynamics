import numpy as np
from shapely.geometry import Polygon, LineString, Point

from crowddynamics.models.game import EgressGame
from crowddynamics.core.motion import Integrator, Adjusting, \
    AgentAgentInteractions, Fluctuation, AgentObstacleInteractions
from crowddynamics.core.navigation import Navigation, Orientation
from crowddynamics.multiagent.simulation import MultiAgentSimulation, TaskNode


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

        self.set_field(domain)

        self.task_graph = TaskNode(Integrator(self, (0.001, 0.01)))
        adjusting = self.task_graph.add_child(Adjusting(self))
        adjusting.add_child(Orientation(self))
        self.task_graph.add_child(AgentAgentInteractions(self))
        self.task_graph.add_child(Fluctuation(self))

        self.set_body(size, body)
        self.set_model(model)
        self.set_agents(**kwargs)


class Hallway(MultiAgentSimulation):
    def __init__(self, queue, size, width, height, model, body):
        super().__init__(queue)
        domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])

        obstacles = (
            LineString([(0, 0), (width, 0)]),
            LineString([(0, height), (width, height)]),
        )

        spawn = (
            Polygon([(1.1, 0),
                     (1.1, height),
                     (width // 2 - 1, height),
                     (width // 2 - 1, 0)]),
            Polygon([(width // 2 + 1, 0),
                     (width // 2 + 1, height),
                     (width - 1.1, height),
                     (width - 1.1, 0)]),
        )

        kwargs = (
            {'size': size // 2,
             'surface': spawn[0],
             'target_direction': np.array((1.0, 0.0)),
             'orientation': 0},
            {'size': size // 2,
             'surface': spawn[1],
             'target_direction': np.array((-1.0, 0.0)),
             'orientation': np.pi},
        )

        self.set_field(domain=domain, obstacles=obstacles)

        self.task_graph = TaskNode(Integrator(self, (0.001, 0.01)))
        adjusting = self.task_graph.add_child(Adjusting(self))
        adjusting.add_child(Orientation(self))
        self.task_graph.add_child(AgentAgentInteractions(self))
        self.task_graph.add_child(AgentObstacleInteractions(self))
        self.task_graph.add_child(Fluctuation(self))

        self.set_body(size, body)
        self.set_model(model)
        for kw in kwargs:
            self.set_agents(**kw)


class Rounding(MultiAgentSimulation):
    def __init__(self, queue, size, width, height, model, body):
        super().__init__(queue)

        domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
        exits = LineString([(0, height / 2), (0, height)])
        obstacles = [
            LineString([(0, height / 2), (width * 3 / 4, height / 2)]),
            domain.exterior - exits
        ]

        spawn = Polygon([(0, 0),
                         (0, height / 2),
                         (width / 2, height / 2),
                         (width / 2, 0)])
        kwargs = {
            'size': size,
            'surface': spawn,
            'target_direction': None,
            'orientation': 0
        }

        self.set_field(domain, obstacles, exits)

        self.task_graph = TaskNode(Integrator(self, (0.001, 0.01)))
        adjusting = self.task_graph.add_child(Adjusting(self))
        adjusting.add_child(Orientation(self))
        adjusting.add_child(Navigation(self))
        self.task_graph.add_child(AgentAgentInteractions(self))
        self.task_graph.add_child(AgentObstacleInteractions(self))
        self.task_graph.add_child(Fluctuation(self))

        self.set_body(size, body)
        self.set_model(model)
        self.set_agents(**kwargs)


class RoomEvacuation(MultiAgentSimulation):
    def __init__(self, queue, size, width, height, model, body, spawn_shape,
                 door_width, exit_hall_width):
        super(RoomEvacuation, self).__init__(queue)

        self.room = Polygon(
            [(0, 0), (0, height), (width, height), (width, 0), ])
        self.hall = Polygon([(width, (height - door_width) / 2),
                             (width, (height + door_width) / 2),
                             (width + exit_hall_width,
                              (height + door_width) / 2),
                             (width + exit_hall_width,
                              (height - door_width) / 2), ])
        self.door = np.array(([(width, (height - door_width) / 2),
                               (width, (height + door_width) / 2), ]))

        exits = LineString([
            (width + exit_hall_width, (height - door_width) / 2),
            (width + exit_hall_width, (height + door_width) / 2),
        ])
        domain = self.room | self.hall
        obstacles = (self.room | self.hall).exterior - exits

        spawn = self.room
        if spawn_shape == "circ":
            spawn = self.room & Point((width, height / 2)).buffer(height / 2)

        kwargs = {
            'size': size,
            'surface': spawn,
            'target_direction': None,
            'orientation': 0
        }

        self.set_field(domain=domain, obstacles=obstacles, exits=exits)

        self.task_graph = TaskNode(Integrator(self, (0.001, 0.01)))
        adjusting = self.task_graph.add_child(Adjusting(self))
        adjusting.add_child(Orientation(self))
        adjusting.add_child(Navigation(self))
        self.task_graph.add_child(AgentAgentInteractions(self))
        self.task_graph.add_child(AgentObstacleInteractions(self))
        self.task_graph.add_child(Fluctuation(self))

        self.set_body(size, body)
        self.set_model(model)
        self.set_agents(**kwargs)


class RoomEvacuationGame(RoomEvacuation):
    def __init__(self, queue, size, width, height, model, body, spawn_shape,
                 door_width, exit_hall_width, t_aset_0, interval,
                 neighbor_radius, neighborhood_size):
        super(RoomEvacuationGame, self).__init__(
            queue, size, width, height, model, body, spawn_shape, door_width,
            exit_hall_width)

        # FIXME
        game = EgressGame(self, self.door, self.room, t_aset_0, interval,
                               neighbor_radius, neighborhood_size)

        self.task_graph = TaskNode(Integrator(self, (0.001, 0.01)))
        adjusting = self.task_graph.add_child(Adjusting(self))
        adjusting.add_child(Orientation(self))
        adjusting.add_child(Navigation(self))
        agent_agent = self.task_graph.add_child(AgentAgentInteractions(self))
        agent_agent.add_child(game)
        self.task_graph.add_child(AgentObstacleInteractions(self))
        self.task_graph.add_child(Fluctuation(self))
