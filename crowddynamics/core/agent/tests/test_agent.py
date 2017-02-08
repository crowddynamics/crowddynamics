import numpy as np
from hypothesis import given
from hypothesis.strategies import data

import crowddynamics.testing.strategies as st
from crowddynamics.core.agent.agent import positions_vector, \
    positions, positions_scalar, Agent


def add_agent(agent, data):
    return agent.add(
        data.draw(st.real(-100, 100, shape=2)),
        data.draw(st.real(1.0, 100.0)),
        data.draw(st.real(0.1, 1.0)),
        data.draw(st.real(0.1, 1.0)),
        data.draw(st.real(0.1, 1.0)),
        data.draw(st.real(0.1, 1.0)),
        data.draw(st.real()),
        data.draw(st.real()),
        data.draw(st.real(-10, 10)),
    )


@given(data())
def test_positions(data):
    # scalar
    out = positions_scalar(
        position=data.draw(st.real(-10.0, 10.0, shape=2)),
        orientation=data.draw(st.real(-np.pi, np.pi)),
        radius_ts=data.draw(st.real(0.01, 0.1))
    )
    assert isinstance(out, tuple)
    for i in range(3):
        arr = out[i]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2,)

    # vector
    size = 2
    out = positions_vector(
        position=data.draw(st.real(-10.0, 10.0, shape=(size, 2))),
        orientation=data.draw(st.real(-np.pi, np.pi, shape=size)),
        radius_ts=data.draw(st.real(0.01, 0.1, shape=size))
    )
    assert isinstance(out, tuple)
    for i in range(3):
        arr = out[i]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (size, 2)

    # scalar
    out = positions(
        position=data.draw(st.real(-10.0, 10.0, shape=2)),
        orientation=data.draw(st.real(-np.pi, np.pi)),
        radius_ts=data.draw(st.real(0.01, 0.1))
    )
    assert isinstance(out, tuple)
    for i in range(3):
        arr = out[i]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2,)

    # vector
    size = 2
    out = positions(
        position=data.draw(st.real(-10.0, 10.0, shape=(size, 2))),
        orientation=data.draw(st.real(-np.pi, np.pi, shape=size)),
        radius_ts=data.draw(st.real(0.01, 0.1, shape=size))
    )
    assert isinstance(out, tuple)
    for i in range(3):
        arr = out[i]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (size, 2)


@given(data())
def test_agent(data):
    size = 10
    agent = Agent(size=size)
    agent.set_circular()
    agent.set_three_circle()

    for indices in range(size):
        index = add_agent(agent, data)
        assert index >= 0
        flag = agent.set_motion(
            index,
            data.draw(st.real(-np.pi, np.pi)),
            data.draw(st.real(-100, 100, shape=2)),
            data.draw(st.real(-100, 100)),
            data.draw(st.unit_vector()),
            data.draw(st.real(-np.pi, np.pi))
        )
        assert flag

    indices = agent.indices()
    assert isinstance(indices, np.ndarray)
    assert indices.dtype.type is np.int64

    index = add_agent(agent, data)
    assert index == -1

    agent.reset_motion()

    agent.remove(0)
    index = add_agent(agent, data)
    assert index >= 0

    out = agent.positions(0)
    assert isinstance(out, tuple)
    assert len(out) == 3
    for indices in range(3):
        arr = out[indices]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2,)
        assert arr.dtype.type is np.float64

    out = agent.radii(0)
    assert isinstance(out, tuple)
    assert len(out) == 3
    for indices in range(3):
        arr = out[indices]
        assert isinstance(arr, float)

    out = agent.front(0)
    assert isinstance(out, np.ndarray)
