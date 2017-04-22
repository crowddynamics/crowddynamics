import numpy as np
from crowddynamics.core.random.sampling import polygon_sample
from crowddynamics.core.structures.agents import Agents, AgentModels
from crowddynamics.core.vector import unit_vector
from crowddynamics.simulation.multiagent import MultiAgentSimulation, \
    Orientation, Integrator, Fluctuation, Adjusting, \
    AgentAgentInteractions, Reset, AgentObstacleInteractions, Navigation
from loggingtools import log_with
from shapely.geometry import Polygon, LineString


@log_with()
def outdoor(size: (1, None) = 100,
            width: (1.0, None) = 20.0,
            height: (1.0, None) = 20.0,
            agent_type: AgentModels = 'circular',
            body_type='adult'):
    """Outdoor

    Simulation for testing collision avoidance generally.

    - Multi-directional flow
    - Periodic boundaries

    Args:
        size: 
        width: 
        height: 
        agent_type: 
        body_type: 
    """
    domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])

    simu = MultiAgentSimulation()
    simu.name = 'Outdoor'

    simu.domain = domain

    simu.agents = Agents(size, agent_type)
    simu.agents.fill(size, {
        'body_type': body_type,
        'position': polygon_sample(np.asarray(domain.exterior)),
        'orientation': lambda: np.random.uniform(-np.pi, np.pi),
        'velocity': lambda: np.random.uniform(0.0, 1.0, 2),
        'angular_velocity': lambda: np.random.uniform(-1.0, 1.0),
        'target_direction': lambda: unit_vector(
            np.random.uniform(-np.pi, np.pi)),
        'target_orientation': lambda: np.random.uniform(-np.pi, np.pi)
    })
    simu.tasks = \
        Reset(simu) << (
            Integrator(simu) << (
                Fluctuation(simu),
                Adjusting(simu) << Orientation(simu),
                AgentAgentInteractions(simu),
            )
        )
    return simu


@log_with()
def hallway(size: (1, None) = 100,
            width: (1.0, None) = 20.0,
            height: (1.0, None) = 5.0,
            agent_type: AgentModels = 'circular',
            body_type='adult'):
    r"""Hallway

    - Low / medium / high crowd density
    - Overtaking
    - Counterflow

    Variables

    #) Bidirectional flow
    #) Unidirectional flow
    """
    domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
    obstacles = LineString([(0, 0), (width, 0)]) | \
                LineString([(0, height), (width, height)])

    spawn1 = Polygon([(1.1, 0),
                      (1.1, height),
                      (width // 2 - 1, height),
                      (width // 2 - 1, 0)]) - obstacles.buffer(0.3)

    spawn2 = Polygon([(width // 2 + 1, 0),
                      (width // 2 + 1, height),
                      (width - 1.1, height),
                      (width - 1.1, 0)]) - obstacles.buffer(0.3)

    simu = MultiAgentSimulation()
    simu.name = 'Hallway'

    simu.domain = domain
    simu.obstacles = obstacles

    simu.agents = Agents(size, agent_type)
    simu.agents.fill(size // 2, {
        'body_type': body_type,
        'position': polygon_sample(np.asarray(spawn1.exterior)),
        'orientation': 0.0,
        'velocity': lambda: np.random.uniform(0.0, 1.0, 2),
        'angular_velocity': 0.0,
        'target_direction': np.array((1.0, 0.0)),
        'target_orientation': 0.0
    })
    simu.agents.fill(size // 2, {
        'body_type': body_type,
        'position': polygon_sample(np.asarray(spawn2.exterior)),
        'orientation': np.pi,
        'velocity': lambda: np.random.uniform(0.0, 1.0, 2),
        'angular_velocity': 0.0,
        'target_direction': np.array((-1.0, 0.0)),
        'target_orientation': np.pi
    })
    simu.tasks = \
        Reset(simu) << (
            Integrator(simu) << (
                Fluctuation(simu),
                Adjusting(simu) << Orientation(simu),
                AgentAgentInteractions(simu),
                AgentObstacleInteractions(simu)
            )
        )
    return simu


@log_with()
def rounding(size: (1, None) = 100,
             width: (1.0, None) = 15.0,
             height: (1.0, None) = 15.0,
             agent_type: AgentModels = 'circular',
             body_type='adult'):
    r"""Rounding

    Simulation for testing navigation algorithm.

    - Unidirectional flow
    """
    domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
    exits = LineString([(0, height / 2), (0, height)])
    obstacles = LineString([(0, height / 2), (width * 3 / 4, height / 2)]) | \
                domain.exterior - exits
    spawn = Polygon([(0, 0),
                     (0, height / 2),
                     (width / 2, height / 2),
                     (width / 2, 0)])

    simu = MultiAgentSimulation()
    simu.name = 'Rounding'

    simu.domain = domain
    simu.obstacles = obstacles
    simu.targets = exits

    simu.agents = Agents(size, agent_type)
    simu.agents.fill(size, {
        'body_type': body_type,
        'position': polygon_sample(np.asarray(spawn.exterior)),
        'orientation': 0.0,
        'velocity': lambda: np.random.uniform(0.0, 1.0, 2),
        'angular_velocity': 0.0,
        'target_direction': np.array((1.0, 0.0)),
        'target_orientation': 0.0
    })
    simu.tasks = \
        Reset(simu) << (
            Integrator(simu) << (
                Fluctuation(simu),
                Adjusting(simu) << (
                    Navigation(simu),
                    Orientation(simu)
                ),
                AgentAgentInteractions(simu),
                AgentObstacleInteractions(simu)
            )
        )
    return simu


@log_with()
def room_evacuation(size: (1, None) = 100,
                    width: (1.0, None) = 10.0,
                    height: (1.0, None) = 20.0,
                    agent_type: AgentModels = 'circular',
                    body_type='adult',
                    door_width: (0.0, None) = 1.2,
                    exit_hall_width: (0.0, None) = 2.0):
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
    room = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
    hall = Polygon([(width, (height - door_width) / 2),
                    (width, (height + door_width) / 2),
                    (width + exit_hall_width, (height + door_width) / 2),
                    (width + exit_hall_width, (height - door_width) / 2)])
    targets = LineString([(width + exit_hall_width, (height - door_width) / 2),
                          (width + exit_hall_width, (height + door_width) / 2)])
    domain = room | hall
    obstacles = (room | hall).exterior - targets

    simu = MultiAgentSimulation()
    simu.name = 'RoomEvacuation'

    simu.domain = domain
    simu.obstacles = obstacles
    simu.targets = targets

    simu.agents = Agents(size, agent_type)
    simu.agents.fill(size, {
        'body_type': body_type,
        'position': polygon_sample(
            np.asarray((room - obstacles.buffer(0.3)).exterior)),
        'orientation': 0.0,
        'velocity': lambda: np.random.uniform(0.0, 1.0, 2),
        'angular_velocity': 0.0,
        'target_direction': np.array((1.0, 0.0)),
        'target_orientation': 0.0
    })
    simu.tasks = \
        Reset(simu) << (
            Integrator(simu) << (
                Fluctuation(simu),
                Adjusting(simu) << (
                    Navigation(simu),
                    Orientation(simu)
                ),
                AgentAgentInteractions(simu),
                AgentObstacleInteractions(simu)
            )
        )
    return simu


@log_with()
def uturn(size: (1, None) = 10,
          width: (1.0, None) = 20.0,
          height: (1.0, None) = 10.0,
          agent_type: AgentModels = 'circular',
          body_type='adult'):
    """U-Turn"""
    domain = Polygon([
        (0, -height / 2), (0, height / 2),
        (width, height / 2), (width, -height / 2)
    ])
    b = 0.9 * height / 2
    b2 = 0.2 * height / 2
    exits = LineString([(0.0, b2), (0.0, b)])
    obstacles = domain - \
                LineString([(0, 0), (0.95 * (width - b), 0)]).buffer(b) | \
                LineString([(0, 0), (0.95 * (width - b), 0)]).buffer(b2) & \
                domain

    spawn = Polygon([(0, -b2), (0, -b),
                     (width / 4, -b), (width / 4, -b2)])

    simu = MultiAgentSimulation()
    simu.name = 'U-Turn'

    simu.domain = domain
    simu.obstacles = obstacles
    simu.targets = exits

    simu.agents = Agents(size, agent_type)
    simu.agents.fill(size, {
        'body_type': body_type,
        'position': polygon_sample(np.asarray(spawn.exterior)),
        'orientation': 0.0,
        'velocity': lambda: np.random.uniform(0.0, 1.0, 2),
        'angular_velocity': 0.0,
        'target_direction': np.array((1.0, 0.0)),
        'target_orientation': 0.0
    })
    simu.tasks = \
        Reset(simu) << (
            Integrator(simu) << (
                Fluctuation(simu),
                Adjusting(simu) << (
                    Navigation(simu),
                    Orientation(simu)
                ),
                AgentAgentInteractions(simu),
                AgentObstacleInteractions(simu)
            )
        )
    return simu


def turning():
    pass


def crossing():
    pass
