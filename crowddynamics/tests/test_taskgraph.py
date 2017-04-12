import random

import hypothesis.strategies as st
from crowddynamics.taskgraph import Node


def newnode(index=0):
    while True:
        yield Node(str(index))
        index += 1


@st.composite
def tree_strategy(draw, maxsize_st=st.integers(1, 100)):
    """
    Search strategy that task graphs.

    Args:
        draw:
        root:
        maxsize_st:

    Returns:
        SearchStrategy:
    """
    # TODO: Control size and depth
    maxsize = draw(maxsize_st)
    nodes_gen = newnode(0)

    size = 1
    depth = 1
    max_leaves = 4

    root = next(nodes_gen)
    parents = {root: 0}
    children = {}

    while size < maxsize:
        leaves = st.integers(1, max_leaves * len(parents))
        for _ in range(draw(leaves)):
            if not size < maxsize:
                break
            parent = random.choice(list(parents.keys()))
            node = next(nodes_gen)
            parent.add_children(node)
            children[node] = 0
            size += 1
            parents[parent] += 1
            if parents[parent] == max_leaves:
                parents.pop(parent)

        parents = children
        children = {}
        depth += 1

    # event("size: {}".format(size))
    # event("depth: {}".format(depth))
    return root


def test_tasknode():
    node = [Node(str(i)) for i in range(4)]
    tree = node[0] << node[1] << (
        node[2],
        node[3]
    )
    assert node[0].is_root
    assert node[0].children == (node[1], node[2], node[3])
