from hypothesis import given, assume, settings

import crowddynamics.testing
from crowddynamics.core.integrator import adaptive_timestep, \
    euler_integration
from crowddynamics.core.integrator.integrator import velocity_verlet


@given(
    dt_min=crowddynamics.testing.real(min_value=0, max_value=10.0),
    dt_max=crowddynamics.testing.real(min_value=0, max_value=10.0),
    velocity=crowddynamics.testing.real(-10, 10, shape=(10, 2)),
    target_velocity=crowddynamics.testing.real(-10, 10, shape=(10, 2))
)
def test_adaptive_timestep(dt_min, dt_max, velocity, target_velocity):
    assume(0 < dt_min < dt_max)
    dt = adaptive_timestep(dt_min, dt_max, velocity, target_velocity)
    assert isinstance(dt, float)
    assert 0 < dt_min <= dt <= dt_max


#TODO: agent strategy
# @given(
#     agent=crowddynamics.testing.agent(size=4),
#     dt_min=crowddynamics.testing.real(min_value=0, max_value=10.0),
#     dt_max=crowddynamics.testing.real(min_value=0, max_value=10.0),
# )
# @settings(max_iterations=5000)
# def test_euler_integration(agent, dt_min, dt_max):
#     assume(0 < dt_min < dt_max)
#     dt = euler_integration(agent, dt_min, dt_max)
#     assert isinstance(dt, float)
#     assert 0 < dt_min <= dt <= dt_max
#
#
# @given(
#     agent=crowddynamics.testing.agent(size=4),
#     dt_min=crowddynamics.testing.real(min_value=0, max_value=10.0),
#     dt_max=crowddynamics.testing.real(min_value=0, max_value=10.0),
# )
# def test_velocity_verlet(agent, dt_min, dt_max):
#     assume(0 < dt_min < dt_max)
#     integrator = velocity_verlet(agent, dt_min, dt_max)
#     for i in range(10):
#         dt = next(integrator)
#         assert isinstance(dt, float)
#         assert 0 < dt_min <= dt <= dt_max
