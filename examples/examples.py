import numpy as np
from loggingtools import log_with

import examples.fields as fields
from crowddynamics.core.structures.agents import Agents, AgentModels
from crowddynamics.core.vector2D import unit_vector
from crowddynamics.simulation.multiagent import MultiAgentSimulation, \
    Orientation, Integrator, Fluctuation, Adjusting, \
    AgentAgentInteractions, Reset, AgentObstacleInteractions, Navigation, \
    InsideDomain


@log_with()
def outdoor(size: (1, None) = 100,
            width: (1.0, None) = 20.0,
            height: (1.0, None) = 20.0,
            agent_type: AgentModels = 'circular',
            body_type='adult'):
    """Simulation for visualizing collision avoidance."""
    simu = MultiAgentSimulation('Outdoor')
    simu.field = fields.outdoor(width, height)
    simu.agents = Agents(size, agent_type)
    simu.agents.add_group(size, {
        'body_type': body_type,
        'position': simu.field.sample_spawn(0),
        'orientation': lambda: np.random.uniform(-np.pi, np.pi),
        'velocity': np.zeros(2),
        'angular_velocity': 0.0,
        'target_direction':
            lambda: unit_vector(np.random.uniform(-np.pi, np.pi)),
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
def hallway(size: (2, None) = 100,
            width: (1.0, None) = 40.0,
            height: (1.0, None) = 5.0,
            agent_type: AgentModels = 'circular',
            body_type='adult'):
    r"""Hallway

    - Low / medium / high crowd density
    - Overtaking
    - Counterflow
    - Bidirectional flow
    - Unidirectional flow

    """
    simu = MultiAgentSimulation('Hallway')
    simu.field = fields.hallway(width, height)
    simu.agents = Agents(size, agent_type)
    simu.agents.add_group(size // 2, {
        'body_type': body_type,
        'position': simu.field.sample_spawn(0),
        'orientation': 0.0,
        'velocity': np.zeros(2),
        'angular_velocity': 0.0,
        'target_direction': np.array((1.0, 0.0)),
        'target_orientation': 0.0,
        'target': 1,
    })
    simu.agents.add_group(size // 2, {
        'body_type': body_type,
        'position': simu.field.sample_spawn(1),
        'orientation': np.pi,
        'velocity': np.zeros(2),
        'angular_velocity': 0.0,
        'target_direction': np.array((-1.0, 0.0)),
        'target_orientation': np.pi,
        'target': 0,
    })
    simu.tasks = \
        Reset(simu) << \
        InsideDomain(simu) << (
            Integrator(simu) << (
                Fluctuation(simu),
                Adjusting(simu) <<
                (Navigation(simu),
                 Orientation(simu)),
                AgentAgentInteractions(simu),
                AgentObstacleInteractions(simu)
            )
        )
    return simu


@log_with()
def rounding(size: (1, None) = 20,
             width: (1.0, None) = 15.0,
             height: (1.0, None) = 15.0,
             agent_type: AgentModels = 'circular',
             body_type='adult'):
    r"""Simulation for testing navigation algorithm.

    - Unidirectional flow
    """
    simu = MultiAgentSimulation('Rounding')
    simu.field = fields.rounding(width, height)
    simu.agents = Agents(size, agent_type)
    simu.agents.add_group(size, {
        'body_type': body_type,
        'position': simu.field.sample_spawn(0),
        'orientation': 0.0,
        'velocity': np.zeros(2),
        'angular_velocity': 0.0,
        'target_direction': np.array((1.0, 0.0)),
        'target_orientation': 0.0,
        'target': 0,
    })
    simu.tasks = \
        Reset(simu) << \
        InsideDomain(simu) << (
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
    simu = MultiAgentSimulation('RoomEvacuation')
    simu.field = fields.room_with_exit(width, height, door_width,
                                       exit_hall_width)
    simu.agents = Agents(size, agent_type)
    simu.agents.add_group(size, {
        'body_type': body_type,
        'position': simu.field.sample_spawn(0),
        'orientation': 0.0,
        'velocity': np.zeros(2),
        'angular_velocity': 0.0,
        'target_direction': np.array((1.0, 0.0)),
        'target_orientation': 0.0,
        'target': 0,
    })
    simu.tasks = \
        Reset(simu) << \
        InsideDomain(simu) << (
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
    simu = MultiAgentSimulation('U-Turn')
    simu.field = fields.uturn(width, height)
    simu.agents = Agents(size, agent_type)
    simu.agents.add_group(size, {
        'body_type': body_type,
        'position': simu.field.sample_spawn(0),
        'orientation': 0.0,
        'velocity': np.zeros(2),
        'angular_velocity': 0.0,
        'target_direction': np.array((1.0, 0.0)),
        'target_orientation': 0.0,
        'target': 0,
    })
    simu.tasks = \
        Reset(simu) << \
        InsideDomain(simu) << (
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


def crossing(size=20,
             l=10,
             d=10,
             u=5,
             v=5,
             k: (0.0, 1.0)=1/2,
             agent_type='circular',
             body_type='adult'):
    """Crossing"""
    simu = MultiAgentSimulation('Crossing')
    simu.field = fields.crossing(l, d, u, v, k)
    simu.agents = Agents(size, agent_type)
    simu.agents.add_group(size // 2, {
        'body_type': body_type,
        'position': simu.field.sample_spawn(0),
        'orientation': 0.0,
        'velocity': np.zeros(2),
        'angular_velocity': 0.0,
        'target_direction': np.array((1.0, 0.0)),
        'target_orientation': 0.0,
        'target': 2,
    })
    simu.agents.add_group(size // 2, {
        'body_type': body_type,
        'position': simu.field.sample_spawn(1),
        'orientation': np.pi / 2,
        'velocity': np.zeros(2),
        'angular_velocity': 0.0,
        'target_direction': np.array((0.0, 1.0)),
        'target_orientation': np.pi / 2,
        'target': 3,
    })
    simu.tasks = \
        Reset(simu) << \
        InsideDomain(simu) << (
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


def bottleneck():
    pass
