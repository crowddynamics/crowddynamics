import random
from tempfile import TemporaryDirectory

import numpy as np

import pytest
from shapely.geometry import Polygon, LineString

from crowddynamics.core.random.sampling import polygon_sample
from crowddynamics.core.structures.agents import AgentModelToType, Agents
from crowddynamics.core.vector import unit_vector
from crowddynamics.simulation.multiagent import MultiAgentSimulation, \
    Integrator, Fluctuation, Adjusting, AgentAgentInteractions, \
    AgentObstacleInteractions, Orientation, Reset, SaveAgentsData, InsideDomain


def samples(spawn, obstacles, radius):
    """Generates positions for agents"""
    geom = spawn - obstacles.buffer(radius)
    vertices = np.asarray(geom.exterior)
    return polygon_sample(vertices)


@pytest.mark.parametrize('size', (1, 2, 10))
@pytest.mark.parametrize('agent_type', tuple(AgentModelToType.values()))
def test_multiagent_simulation(size, agent_type):
    height = 20
    width = 20

    domain = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
    obstacles = LineString([(0, 0), (width, 0)]) | \
                LineString([(0, height), (width, height)])
    spawn = Polygon([(1.1, 0),
                     (1.1, height),
                     (width // 2 - 1, height),
                     (width // 2 - 1, 0)])

    simu = MultiAgentSimulation()
    simu.name = 'Testing {}'.format(agent_type)
    simu.domain = domain
    simu.obstacles = obstacles
    simu.agents = Agents(size, agent_type)
    simu.agents.fill(size, {
        'body_type': lambda: random.choice(
            ('adult', 'male', 'female', 'child', 'eldery')),
        'position': samples(spawn, obstacles, 0.3),
        'orientation': lambda: np.random.uniform(-np.pi, np.pi),
        'velocity': lambda: np.random.uniform(0.0, 1.0, 2),
        'angular_velocity': lambda: np.random.uniform(-1.0, 1.0),
        'target_direction': lambda: unit_vector(np.random.uniform(-np.pi, np.pi)),
        'target_orientation': lambda: np.random.uniform(-np.pi, np.pi)
    })

    with TemporaryDirectory() as tmpdir:
        simu.tasks = \
            Reset(simu) << \
            SaveAgentsData(simu, tmpdir) << \
            InsideDomain(simu) << (
                Integrator(simu) << (
                    Fluctuation(simu),
                    Adjusting(simu) << Orientation(simu),
                    AgentAgentInteractions(simu),
                    AgentObstacleInteractions(simu),
                )
            )
        for _ in range(10):
            simu.update()
