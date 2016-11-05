from crowddynamics.functions import public


@public
class TaskNode:
    """
    Create task graphs. Task graphs are used to create order for function calls
    in simulation and to evaluate functions in this order.
    """

    def __init__(self):
        self._child_nodes = []

    def update(self):
        """
        This method should be implemented by the superclass.
        """
        raise NotImplementedError

    def add_child(self, node):
        """

        Args:
            node: TaskNode

        Returns:
            None

        """
        self._child_nodes.append(node)

    def evaluate(self):
        """
        Evaluate nodes recursively, child nodes first (bottom-up).

        Returns:
            None

        """
        for node in self._child_nodes:
            node.evaluate()
        self.update()

    def __iadd__(self, other):
        """
        Syntactic sugar for adding new child nodes.

        Args:
            other: TaskNode

        Returns:
            None

        """
        self.add_child(other)
        return self


class RootNode(TaskNode):
    def __init__(self):
        super().__init__()

    def update(self):
        pass
