"""Example multiagent simulations

Todo:
    - Periodic boundaries
    - Convert examples into test and validation simulations
    - Move simulations.yaml configuration parameter into the classes
"""
import numpy as np
from shapely.geometry import Polygon, LineString, Point

from crowddynamics.logging import log_with
from crowddynamics.multiagent.simulation import MultiAgentSimulation, register, \
    AGENT_MODELS, BODY_TYPES
from crowddynamics.multiagent.tasks import Navigation, Orientation, \
    Integrator, Fluctuation, Adjusting, AgentAgentInteractions, \
    AgentObstacleInteractions, Reset


class Outdoor(MultiAgentSimulation):
    r"""Outdoor

    Simulation for testing collision avoidance generally.

    - Multi-directional flow
    - Periodic boundaries

    """

    @log_with()
    def set(self,
            size: (1, None) = 100,
            width: (1.0, None) = 20.0,
            height: (1.0, None) = 20.0,
            model: AGENT_MODELS = 'three_circle',
            body_type: BODY_TYPES = 'adult'):

        reset = Reset(self)
        integrator = Integrator(self)
        reset += integrator
        adjusting = Adjusting(self)
        adjusting += Orientation(self)
        integrator += adjusting
        integrator += AgentAgentInteractions(self)
        integrator += Fluctuation(self)
        self.set_tasks(reset)

        domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
        self.init_domain(domain)

        self.init_agents(size, model)
        for i in self.add_agents(size, self.domain, body_type):
            pass


class Hallway(MultiAgentSimulation):
    r"""Hallway

    - Low / medium / high crowd density
    - Overtaking
    - Counterflow

    Variables

    #) Bidirectional flow
    #) Unidirectional flow
    """

    @log_with()
    def set(self,
            size: (1, None) = 100,
            width: (1.0, None) = 20.0,
            height: (1.0, None) = 5.0,
            model: AGENT_MODELS = 'three_circle',
            body_type: BODY_TYPES = 'adult'):

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

        self.set_tasks(reset)

        for kw in kwargs:
            for i in self.add_agents(kw['size'], kw['spawn'], body_type):
                self.agent.set_motion(
                    i, kw['orientation'], np.zeros(2), 0.0,
                    kw['target_direction'], kw['orientation']
                )


class Crossing(MultiAgentSimulation):
    r"""Crossing

    - Orthogonal flow
    """

    @log_with()
    def set(self, *args, **kwargs):
        pass


class Rounding(MultiAgentSimulation):
    r"""Rounding

    Simulation for testing navigation algorithm.

    - Unidirectional flow
    """

    @log_with()
    def set(self,
            size: (1, None) = 100,
            width: (1.0, None) = 15.0,
            height: (1.0, None) = 15.0,
            model: AGENT_MODELS = 'circular',
            body_type: BODY_TYPES = 'adult'):

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

        self.set_tasks(reset)

        for kw in kwargs:
            for i in self.add_agents(kw['size'], kw['spawn'], body_type):
                self.agent.set_motion(
                    i, kw['orientation'], np.zeros(2), 0.0,
                    kw['target_direction'], kw['orientation']
                )


class RoomEvacuation(MultiAgentSimulation):
    r"""Room Evacuation

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

    @log_with()
    def set(self,
            size: (1, None) = 100,
            width: (1.0, None) = 10.0,
            height: (1.0, None) = 20.0,
            model: AGENT_MODELS = 'circular',
            body_type: BODY_TYPES = 'adult',
            spawn_shape: ('circ',) = 'circ',
            door_width: (0.5, None) = 1.2,
            exit_hall_width: (0.0, None) = 2.0):

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

        self.set_tasks(reset)

        for kw in kwargs:
            for i in self.add_agents(kw['size'], kw['spawn'], body_type):
                self.agent.set_motion(
                    i, kw['orientation'], np.zeros(2), 0.0,
                    kw['target_direction'], kw['orientation']
                )


def init():
    register(Outdoor)
    register(Hallway)
    # register(Crossing)
    register(Rounding)
    # register(RoomEvacuation)
