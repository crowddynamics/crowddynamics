from hypothesis import given

import crowddynamics.testing.strategies as st
from crowddynamics.multiagent import Agent


def test_agent():
    agent = Agent(size=10)
    assert True


@given(
    position=st.vector(),
    mass=st.real(0.0),
    radius=st.real(0.0),
    ratio_rt=st.real(0.0, 1.0),
    ratio_rs=st.real(0.0, 1.0),
    ratio_ts=st.real(0.0, 1.0),
    inertia_rot=st.real(0.0),
    max_velocity=st.real(0.0),
    max_angular_velocity=st.real(0.0)
)
def test_agent_add(position, mass, radius, ratio_rt, ratio_rs, ratio_ts,
                   inertia_rot, max_velocity, max_angular_velocity):
    agent = Agent(size=10)
    success = agent.add(position, mass, radius, ratio_rt, ratio_rs, ratio_ts,
                        inertia_rot, max_velocity, max_angular_velocity)
    assert True
