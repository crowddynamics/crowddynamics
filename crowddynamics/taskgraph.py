"""Task Graph

https://www.python.org/doc/essays/graphs/
"""


class TaskNode:
    """TaskNode

    Create task graphs with TaskNodes for evaluating simulation algorithms
    in the right order and possibly parallel.
    """

    def __init__(self):
        # TODO: hashing, __getitem__
        self._children = []

    def update(self, *args, **kwargs):
        """Method that is called when the node is evaluated. Overridden by super
        classes."""
        pass

    def evaluate(self, *args, **kwargs):
        """Evaluate nodes recursively, child nodes first (bottom-up)."""
        for node in self._children:
            node.evaluate()
        self.update()

    def add(self, node):
        """
        Add new child node.

        Args:
            node: TaskNode
        """
        self._children.append(node)

    def __iadd__(self, other):
        """
        Syntactic sugar for adding new child nodes.

        Args:
            other: TaskNode

        Returns:
            TaskNode: self
        """
        self.add(other)
        return self


class TaskNodeGroup:
    pass


class RootNode:
    pass
