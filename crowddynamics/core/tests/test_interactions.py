import hypothesis.strategies as st
from hypothesis.core import given
from hypothesis.extra.numpy import arrays

import crowddynamics.testing as testing
from crowddynamics.core.interactions import (
    interaction_agent_agent_circular,
    interaction_agent_agent_three_circle,
    agent_agent_block_list,
    agent_circular_obstacle, agent_three_circle_obstacle)
from crowddynamics.core.structures import obstacle_type_linear
from crowddynamics.simulation.agents import Circular, ThreeCircle

CELL_SIZE = 3.6
agent_attributes = {
    'radius': testing.reals(0, 1.0, exclude_zero='near'),
    'mass': testing.reals(0, 1.0, exclude_zero='near'),
    'positions': testing.reals(-10, 10, shape=2),
    'velocity': testing.reals(-1, 1, shape=2),
    'body_type': st.just('adult')
}


# Agent-agent

@given(testing.agents(size_strategy=st.just(2),
                      agent_type=Circular,
                      attributes=agent_attributes))
def test_agent_interactions_circular(agents):
    interaction_agent_agent_circular(0, 1, agents)
    assert True


@given(testing.agents(size_strategy=st.just(2),
                      agent_type=ThreeCircle,
                      attributes=agent_attributes))
def test_agent_interactions_three_circle(agents):
    interaction_agent_agent_three_circle(0, 1, agents)
    assert True


@given(testing.agents(size_strategy=st.integers(0, 2),
                      agent_type=Circular,
                      attributes=agent_attributes))
def test_agent_block_list_circular(agents):
    agent_agent_block_list(agents, CELL_SIZE)
    assert True


@given(testing.agents(size_strategy=st.integers(0, 2),
                      agent_type=ThreeCircle,
                      attributes=agent_attributes))
def test_agent_block_list_three_circle(agents):
    agent_agent_block_list(agents, CELL_SIZE)
    assert True


# Agent-obstacle

@given(agents=testing.agents(size_strategy=st.just(1),
                             agent_type=Circular,
                             attributes=agent_attributes),
       obstacles=arrays(dtype=obstacle_type_linear, shape=1,
                        elements=st.tuples(testing.reals(-10, 10),
                                           testing.reals(-10, 10))))
def test_agent_circular_obstacle(agents, obstacles):
    agent_circular_obstacle(agents, obstacles)
    assert True


@given(agents=testing.agents(size_strategy=st.just(1),
                             agent_type=ThreeCircle,
                             attributes=agent_attributes),
       obstacles=arrays(dtype=obstacle_type_linear, shape=1,
                        elements=st.tuples(testing.reals(-10, 10),
                                           testing.reals(-10, 10))))
def test_agent_three_circle_obstacle(agents, obstacles):
    agent_three_circle_obstacle(agents, obstacles)
    assert True
