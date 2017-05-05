"""Tree node for implementing simulation logic

https://www.python.org/doc/essays/graphs/
http://dask.pydata.org/en/latest/
http://dask.pydata.org/en/latest/graphs.html
https://en.wikipedia.org/wiki/Tree_(data_structure)
https://github.com/c0fec0de/anytree

Todo:
    - update method timer
"""

from anytree import NodeMixin, PreOrderIter


class Node(NodeMixin):
    """Node (tree)
    
    Node for implementing trees for controlling evaluation order for
    simulation logic.
    """

    def __init__(self, name=None, parent=None):
        self._name = name
        self.parent = parent

    @property
    def name(self):
        """Name

        Returns:
            str: Return name set by user. If name is not set returns class
                 name.
        """
        return self._name if self._name else self.__class__.__name__

    @name.setter
    def name(self, value):
        """Name setter"""
        self._name = value

    def update(self, *args, **kwargs):
        """Method that is called when the node is evaluated. Overridden by super
        classes."""
        pass

    def inject_before(self, node):
        """Inject before

        Args:
            node (Node): 
        """
        parent = self.parent
        self.parent = node
        node.parent = parent

    def inject_after(self, node):
        """Inject after

        Args:
            node (Node): 
        """
        for child in self.children:
            child.parent = node
        node.parent = self

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
        if isinstance(other, Node):
            self.add_children(other)
        else:
            for _other in other:
                self.add_children(_other)
        return self

    def __repr__(self):
        return self.__class__.__name__

    def __getitem__(self, item):
        """Slow get item which find node named "item" by iterating over the all 
        nodes."""
        for node in PreOrderIter(self.root):
            if node.name == item:
                return node
        raise KeyError('Key: "{}" not in the tree.'.format(item))
