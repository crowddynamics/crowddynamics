import random

import hypothesis.strategies as st
from hypothesis import event
from hypothesis import given

from crowddynamics.taskgraph import TaskNode


class Node(TaskNode):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def update(self, l):
        l.append(self.index)

    def evaluate(self, l):
        """
        Test evaluate

        Args:
            l (list):

        Returns:
            list:
        """
        for node in self._children:
            node.evaluate(l)
        self.update(l)


def newnode(index=0):
    while True:
        yield Node(index)
        index += 1


@st.composite
def task_graph(draw, maxsize_st=st.integers(1, 100)):
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
            parent += node
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


@given(task_graph())
def test_taskgraph(tasks):
    l = []
    tasks.evaluate(l)
    assert True
