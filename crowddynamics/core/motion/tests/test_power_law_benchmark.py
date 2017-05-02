import pytest
import numpy as np

from crowddynamics.core.motion.power_law import force_social_circular, \
    force_social_three_circle
from crowddynamics.core.structures.agents import Agents, agent_type_circular, \
    agent_type_three_circle
from crowddynamics.core.vector2D import length


@pytest.mark.parametrize('agent_type,force', [
    (agent_type_circular, force_social_circular),
    (agent_type_three_circle, force_social_three_circle)])
def test_not_colliding(benchmark, agent_type, force):
    agents = Agents(2, agent_type)
    agents.add_group(1, {
        'body_type': 'adult',
        'position': np.array((0.0, 0.0)),
        'orientation': 0.0,
        'velocity': np.array((1.0, 0.0)),
        'angular_velocity': 0.0,
        'target_direction': np.array((1.0, 0.0)),
        'target_orientation': 0.0
    })
    agents.add_group(1, {
        'body_type': 'adult',
        'position': np.array((2.0, 0.0)),
        'orientation': 0.0,
        'velocity': np.array((1.0, 0.0)),
        'angular_velocity': 0.0,
        'target_direction': np.array((1.0, 0.0)),
        'target_orientation': 0.0
    })
    force_i, force_j = benchmark(force, agents.array, 0, 1)
    assert length(force_i) == 0
    assert length(force_j) == 0


@pytest.mark.parametrize('agent_type,force', [
    (agent_type_circular, force_social_circular),
    (agent_type_three_circle, force_social_three_circle)])
def test_colliding(benchmark, agent_type, force):
    agents = Agents(2, agent_type)
    agents.add_group(1, {
        'body_type': 'adult',
        'position': np.array((0.0, 0.0)),
        'orientation': 0.0,
        'velocity': np.array((1.0, 0.0)),
        'angular_velocity': 0.0,
        'target_direction': np.array((1.0, 0.0)),
        'target_orientation': 0.0
    })
    agents.add_group(1, {
        'body_type': 'adult',
        'position': np.array((2.0, 0.0)),
        'orientation': np.pi,
        'velocity': np.array((-1.0, 0.0)),
        'angular_velocity': 0.0,
        'target_direction': np.array((-1.0, 0.0)),
        'target_orientation': np.pi
    })
    force_i, force_j = benchmark(force, agents.array, 0, 1)
    assert length(force_i) > 0
    assert length(force_j) > 0
