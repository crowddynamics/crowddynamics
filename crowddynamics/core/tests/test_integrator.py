from hypothesis import given, assume

from crowddynamics.core.integrator import adaptive_timestep, \
    euler_integrator, velocity_verlet_integrator
from crowddynamics.testing import reals


@given(dt_min=reals(min_value=0, max_value=10.0, exclude_zero='near'),
       dt_max=reals(min_value=0, max_value=10.0, exclude_zero='near'))
def test_adaptive_timestep1(agents_circular, dt_min, dt_max):
    assume(0 < dt_min < dt_max)
    dt = adaptive_timestep(agents_circular.array, dt_min, dt_max)
    assert 0 < dt_min <= dt <= dt_max


@given(dt_min=reals(min_value=0, max_value=10.0, exclude_zero='near'),
       dt_max=reals(min_value=0, max_value=10.0, exclude_zero='near'))
def test_adaptive_timestep2(agents_three_circle, dt_min, dt_max):
    assume(0 < dt_min < dt_max)
    dt = adaptive_timestep(agents_three_circle.array, dt_min, dt_max)
    assert 0 < dt_min <= dt <= dt_max


@given(dt_min=reals(min_value=0, max_value=10.0, exclude_zero='near'),
       dt_max=reals(min_value=0, max_value=10.0, exclude_zero='near'))
def test_euler_integration1(agents_circular, dt_min, dt_max):
    assume(0 < dt_min < dt_max)
    dt = euler_integrator(agents_circular.array, dt_min, dt_max)
    assert 0 < dt_min <= dt <= dt_max


@given(dt_min=reals(min_value=0, max_value=10.0, exclude_zero='near'),
       dt_max=reals(min_value=0, max_value=10.0, exclude_zero='near'))
def test_euler_integration2(agents_three_circle, dt_min, dt_max):
    assume(0 < dt_min < dt_max)
    dt = euler_integrator(agents_three_circle.array, dt_min, dt_max)
    assert 0 < dt_min <= dt <= dt_max


@given(dt_min=reals(min_value=0, max_value=10.0),
       dt_max=reals(min_value=0, max_value=10.0))
def test_velocity_verlet1(agents_circular, dt_min, dt_max):
    assume(0 < dt_min < dt_max)
    dt = velocity_verlet_integrator(agents_circular.array, dt_min, dt_max)
    assert 0 < dt_min <= dt <= dt_max


@given(dt_min=reals(min_value=0, max_value=10.0),
       dt_max=reals(min_value=0, max_value=10.0))
def test_velocity_verlet2(agents_three_circle, dt_min, dt_max):
    assume(0 < dt_min < dt_max)
    dt = velocity_verlet_integrator(agents_three_circle.array, dt_min, dt_max)
    assert 0 < dt_min <= dt <= dt_max
