import pytest
import numpy as np
from hypothesis.core import given
import hypothesis.strategies as st
from hypothesis.extra.numpy import floating_dtypes

from crowddynamics.core.vector2D import length
from crowddynamics.simulation.agents import Circular, ThreeCircle
import crowddynamics.testing as testing


@pytest.mark.parametrize('exclude_zero', (None, 'near', 'exact'))
@given(dtype=st.none() | floating_dtypes(),
       shape=st.none() | st.integers(0, 2) |
             st.tuples(st.integers(0, 2), st.integers(0, 2)))
def test_reals(exclude_zero, shape, dtype):
    strategy = testing.reals(min_value=None,
                             max_value=None,
                             exclude_zero=exclude_zero,
                             shape=shape,
                             dtype=dtype)
    for _ in range(10):
        strategy.example()
    assert True


@given(v=testing.unit_vectors())
def test_unit_vectors(v):
    assert np.isclose(length(v), 1.0)


def test_points():
    strategy = testing.points()
    for _ in range(10):
        strategy.example()
    assert True


@pytest.mark.parametrize('num_verts', (2, 3, 4, 5))
@pytest.mark.parametrize('closed', (False, True))
def test_linestring(num_verts, closed):
    strategy = testing.linestrings(num_verts=num_verts, closed=closed)
    for _ in range(10):
        strategy.example()
    assert True


@pytest.mark.parametrize('num_verts', (3, 4, 5))
@pytest.mark.parametrize('has_holes', (False, True))
def test_polygons(num_verts, has_holes):
    strategy = testing.polygons(num_verts=num_verts, has_holes=has_holes)
    for _ in range(10):
        strategy.example()
    assert True


agent_attributes = {
    'radius': testing.reals(0.1, 1.0),
    'mass': testing.reals(0.1, 1.0),
    'body_type': st.sampled_from(('adult', 'male', 'female'))
}


@pytest.mark.parametrize('agent_type, attributes',
                         [(Circular, agent_attributes),
                          (ThreeCircle, agent_attributes)])
def test_agents(agent_type, attributes):
    agents = testing.agents(size_strategy=st.just(10),
                            agent_type=agent_type,
                            attributes=attributes)
    assert agents.example().shape == (10,)


def test_obstacles():
    assert True
