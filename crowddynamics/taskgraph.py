"""
Task Graph
"""


class TaskNode:
    """
    Create task graphs. Task graphs are used to create order for function calls
    in simulation and to evaluate functions in this order.
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
