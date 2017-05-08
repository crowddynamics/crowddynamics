import numpy as np
import pytest

from crowddynamics.core.interactions.interactions import \
    agent_agent_interaction_circle, agent_agent_interaction_three_circle, \
    agent_agent_block_list
from crowddynamics.core.interactions.interactions_mt import \
    agent_agent_block_list_multithreaded


def test_agent_agent_interaction_circle(agents_circular):
    agent_agent_interaction_circle(0, 1, agents_circular.array)
    assert True


def test_agent_agent_interaction_three_circle(agents_three_circle):
    agent_agent_interaction_three_circle(0, 1, agents_three_circle.array)
    assert True


def test_agent_agent_block_list_circular(agents_circular):
    agent_agent_block_list(agents_circular.array)
    assert True


def test_agent_agent_block_list_three_circle(agents_three_circle):
    agent_agent_block_list(agents_three_circle.array)
    assert True


def test_agent_agent_block_list_circular_multithreaded(agents_circular):
    agent_agent_block_list_multithreaded(agents_circular.array)
    assert True


def test_agent_agent_block_list_three_circle_multithreaded(agents_three_circle):
    agent_agent_block_list_multithreaded(agents_three_circle.array)
    assert True


def test_agent_agent_block_list_circular_cmp(agents_circular):
    a = np.copy(agents_circular.array)
    b = np.copy(agents_circular.array)

    agent_agent_block_list(a)
    agent_agent_block_list_multithreaded(b)

    assert np.all(a == b)


def test_agent_agent_block_list_three_circle_cmp(agents_three_circle):
    a = np.copy(agents_three_circle.array)
    b = np.copy(agents_three_circle.array)

    agent_agent_block_list(a)
    agent_agent_block_list_multithreaded(b)

    assert np.all(a == b)
