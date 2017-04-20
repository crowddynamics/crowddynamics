from crowddynamics.core.interactions.interactions import \
    agent_agent_interaction_circle, agent_agent_interaction_three_circle, \
    agent_agent_block_list_circular, agent_agent_block_list_three_circle


def test_agent_agent_interaction_circle(agents_circular):
    agent_agent_interaction_circle(0, 1, agents_circular.array)
    assert True


def test_agent_agent_interaction_three_circle(agents_three_circle):
    agent_agent_interaction_three_circle(0, 1, agents_three_circle.array)
    assert True


def test_agent_agent_block_list_circular(agents_circular):
    agent_agent_block_list_circular(agents_circular.array)
    assert True


def test_agent_agent_block_list_three_circle(agents_three_circle):
    agent_agent_block_list_three_circle(agents_three_circle.array)
    assert True
