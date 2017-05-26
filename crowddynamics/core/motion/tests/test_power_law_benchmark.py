import numpy as np
import pytest

from crowddynamics.core.motion.power_law import force_social_circular, \
    force_social_three_circle
from crowddynamics.core.vector2D import length
from crowddynamics.simulation.agents import Circular, ThreeCircle


@pytest.mark.parametrize('agent_type,force', [
    (Circular, force_social_circular),
    (ThreeCircle, force_social_three_circle)])
def test_not_colliding(benchmark, agent_type, force):
    agent1 = agent_type(**{
            'body_type': 'adult',
            'position': np.array((0.0, 0.0)),
            'orientation': 0.0,
            'velocity': np.array((1.0, 0.0)),
            'angular_velocity': 0.0,
            'target_direction': np.array((1.0, 0.0)),
            'target_orientation': 0.0
        })
    agent2 = agent_type(**{
            'body_type': 'adult',
            'position': np.array((2.0, 0.0)),
            'orientation': 0.0,
            'velocity': np.array((1.0, 0.0)),
            'angular_velocity': 0.0,
            'target_direction': np.array((1.0, 0.0)),
            'target_orientation': 0.0
        })
    array = np.concatenate((np.array(agent1), np.array(agent2)))
    force_i, force_j = benchmark(force, array, 0, 1)
    assert length(force_i) == 0
    assert length(force_j) == 0


@pytest.mark.parametrize('agent_type,force', [
    (Circular, force_social_circular),
    (ThreeCircle, force_social_three_circle)])
def test_colliding(benchmark, agent_type, force):
    agent1 = agent_type({
        'body_type': 'adult',
        'position': np.array((0.0, 0.0)),
        'orientation': 0.0,
        'velocity': np.array((1.0, 0.0)),
        'angular_velocity': 0.0,
        'target_direction': np.array((1.0, 0.0)),
        'target_orientation': 0.0
    })
    agent2 = agent_type({
        'body_type': 'adult',
        'position': np.array((2.0, 0.0)),
        'orientation': np.pi,
        'velocity': np.array((-1.0, 0.0)),
        'angular_velocity': 0.0,
        'target_direction': np.array((-1.0, 0.0)),
        'target_orientation': np.pi
    })
    array = np.concatenate((np.array(agent1), np.array(agent2)))
    force_i, force_j = benchmark(force, array, 0, 1)
    assert length(force_i) > 0
    assert length(force_j) > 0
