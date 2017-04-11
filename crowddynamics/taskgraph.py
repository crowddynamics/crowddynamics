"""Task Graph

https://www.python.org/doc/essays/graphs/
http://dask.pydata.org/en/latest/
http://dask.pydata.org/en/latest/graphs.html
https://en.wikipedia.org/wiki/Tree_(data_structure)
https://github.com/c0fec0de/anytree
"""
from anytree import NodeMixin


class TaskNode(NodeMixin, object):
    """TaskNode

    Create task graphs with TaskNodes for evaluating simulation algorithms
    in the right order and possibly parallel.
    """

    def __init__(self, name=None, parent=None):
        self._name = name
        self.parent = parent

    @property
    def name(self):
        """Name."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def update(self, *args, **kwargs):
        """Method that is called when the node is evaluated. Overridden by super
        classes."""
        pass

    def evaluate(self, *args, **kwargs):
        """Evaluate nodes recursively, child nodes first (bottom-up)."""
        for node in self._children:
            node.evaluate()
        self.update()

    def add_children(self, node):
        """Add new child node.

        Args:
            node: TaskNode
        """
        node.parent = self
        return self

    def __lshift__(self, other):
        """Syntactic sugar for forming task graphs
        
        ::
        
            simu.tasks = \
                Reset(simu) << (
                    Integrator(simu) << (
                        Fluctuation(simu),
                        Adjusting(simu) << Orientation(simu),
                        AgentAgentInteractions(simu),
                        AgentObstacleInteractions(simu),
                    )
                )
    
        Args:
            other (TaskNode, Tuple[TaskNode]): 
        """
        if isinstance(other, TaskNode):
            self.add_children(other)
        else:
            for _other in other:
                self.add_children(_other)
        return self

    def __repr__(self):
        classname = self.__class__.__name__
        args = ["%r" % self.separator.join([""] + [str(node.name) for node in self.path])]
        for key, value in filter(lambda item: not item[0].startswith("_"),
                                 sorted(self.__dict__.items(),
                                        key=lambda item: item[0])):
            args.append("%s=%r" % (key, value))
        return "%s(%s)" % (classname, ", ".join(args))


class TaskNodeGroup:
    pass
