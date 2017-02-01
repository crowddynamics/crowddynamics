"""Example multiagent simulations

Todo:
    - Periodic boundaries
    - Convert examples into test and validation simulations
    - Move simulations.yaml configuration parameter into the classes
"""
import numpy as np
from shapely.geometry import Polygon, LineString, Point

from crowddynamics.multiagent.simulation import MultiAgentSimulation
from crowddynamics.multiagent.tasks import Navigation, Orientation, \
    Integrator, Fluctuation, Adjusting, AgentAgentInteractions, \
    AgentObstacleInteractions, Reset


class Outdoor(MultiAgentSimulation):
    r"""
    Outdoor

    Simulation for testing collision avoidance generally.

    - Multi-directional flow
    - Periodic boundaries

    Tasks

    - Reset
        - Integrator
            - Adjusting
                - Orientation
            - AgentAgentInteractions
            - Fluctuation
    """
    def __init__(self, queue, size, width, height, model, body_type):
        """
        Init

        Args:
            queue:
            size:
            width:
            height:
            model:
            body_type:

        """
        super().__init__(queue, "Outdoor")

        domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
        self.init_domain(domain)
        self.init_agents(size, model)

        reset = Reset(self)
        integrator = Integrator(self)
        reset += integrator
        adjusting = Adjusting(self)
        adjusting += Orientation(self)
        integrator += adjusting
        integrator += AgentAgentInteractions(self)
        integrator += Fluctuation(self)

        self.tasks += reset

        for i in self.add_agents(size, domain, body_type):
            pass


class Hallway(MultiAgentSimulation):
    r"""
    Hallway

    - Low / medium / high crowd density
    - Overtaking
    - Counterflow

    Variables

    #) Bidirectional flow
    #) Unidirectional flow
    """
    def __init__(self, queue, size, width, height, model, body_type):
        super().__init__(queue, "Hallway")
        domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
        obstacles = (
            LineString([(0, 0), (width, 0)]),
            LineString([(0, height), (width, height)]),
        )
        spawns = (
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
             'spawn': spawns[0],
             'target_direction': np.array((1.0, 0.0)),
             'orientation': 0},
            {'size': size // 2,
             'spawn': spawns[1],
             'target_direction': np.array((-1.0, 0.0)),
             'orientation': np.pi},
        )

        self.init_domain(domain)
        self.init_agents(size, model)

        for obs in obstacles:
            self.add_obstacle(obs)

        reset = Reset(self)
        integrator = Integrator(self, (0.001, 0.01))
        reset += integrator
        adjusting = Adjusting(self)
        adjusting += Orientation(self)
        integrator += adjusting
        integrator += AgentAgentInteractions(self)
        integrator += AgentObstacleInteractions(self)
        integrator += Fluctuation(self)

        self.tasks += reset

        for kw in kwargs:
            for i in self.add_agents(kw['size'], kw['spawn'], body_type):
                self.agent.set_motion(
                    i, kw['orientation'], np.zeros(2), 0.0,
                    kw['target_direction'], kw['orientation']
                )


class Crossing(MultiAgentSimulation):
    r"""
    Crossing

    - Orthogonal flow
    """
    pass


class Rounding(MultiAgentSimulation):
    r"""
    Rounding

    Simulation for testing navigation algorithm.

    - Unidirectional flow
    """
    def __init__(self, queue, size, width, height, model, body_type):
        super().__init__(queue, "Rounding")

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
        kwargs = [{
            'size': size,
            'spawn': spawn,
            'target_direction': np.array((1, 0)),
            'orientation': 0
        }]

        self.init_domain(domain)
        self.init_agents(size, model)
        for obs in obstacles:
            self.add_obstacle(obs)
        self.add_target(exits)

        reset = Reset(self)
        integrator = Integrator(self, (0.001, 0.01))
        reset += integrator
        adjusting = Adjusting(self)
        adjusting += Orientation(self)
        adjusting += Navigation(self)
        integrator += adjusting
        integrator += AgentAgentInteractions(self)
        integrator += AgentObstacleInteractions(self)
        integrator += Fluctuation(self)

        self.tasks += reset

        for kw in kwargs:
            for i in self.add_agents(kw['size'], kw['spawn'], body_type):
                self.agent.set_motion(
                    i, kw['orientation'], np.zeros(2), 0.0,
                    kw['target_direction'], kw['orientation']
                )


class RoomEvacuation(MultiAgentSimulation):
    r"""
    Room Evacuation

    - Unidirectional flow
    - Door width
    - Door capacity
    - Herding
    - Exit selection

    Variables

    - Number of exits
    - One exit
    - Two exits
    - Multiple exits
    """
    def __init__(self, queue, size, width, height, model, body_type,
                 spawn_shape, door_width, exit_hall_width):
        super(RoomEvacuation, self).__init__(queue, "RoomEvacuation")

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

        kwargs = [{
            'size': size,
            'spawn': spawn,
            'target_direction': np.array((1, 0)),
            'orientation': 0
        }]

        self.init_domain(domain)
        self.init_agents(size, model)
        for obs in obstacles:
            self.add_obstacle(obs)
        self.add_target(exits)

        reset = Reset(self)
        integrator = Integrator(self, (0.001, 0.01))
        reset += integrator
        adjusting = Adjusting(self)
        adjusting += Orientation(self)
        adjusting += Navigation(self)
        integrator += adjusting
        integrator += AgentAgentInteractions(self)
        integrator += AgentObstacleInteractions(self)
        integrator += Fluctuation(self)

        self.tasks += reset

        for kw in kwargs:
            for i in self.add_agents(kw['size'], kw['spawn'], body_type):
                self.agent.set_motion(
                    i, kw['orientation'], np.zeros(2), 0.0,
                    kw['target_direction'], kw['orientation']
                )
