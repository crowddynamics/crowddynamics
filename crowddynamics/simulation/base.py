"""Base classes for simulation objects"""
from collections import OrderedDict
from datetime import datetime

from anytree.node import NodeMixin, PreOrderIter
from traitlets.traitlets import HasTraits, Unicode, default, Instance


class CrowdDynamicsObject(HasTraits):
    """Base class for CrowdDynamics simulation objects"""
    _datetime_fmt = '%Y-%m-%d_%H:%M:%S.%f'

    name = Unicode(help='Name for the object. Default to class name.')

    @default('name')
    def _default_name(self):
        return self.__class__.__name__


class FieldBase(CrowdDynamicsObject):
    pass


class AgentsBase(CrowdDynamicsObject):
    pass


class LogicNodeBase(CrowdDynamicsObject, NodeMixin):
    """Node for implementing trees for controlling evaluation order for 
    simulation logic."""

    def update(self, *args, **kwargs):
        """Method that is called when the node is evaluated."""
        raise NotImplementedError

    def inject_before(self, node):
        """Inject before"""
        parent = self.parent
        self.parent = node
        node.parent = parent

    def inject_after(self, node):
        """Inject after"""
        for child in self.children:
            child.parent = node
        node.parent = self

    def add_children(self, node):
        """Add new child node."""
        node.parent = self
        return self

    def __lshift__(self, other):
        """Syntactic sugar for forming task graphs::

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
        if isinstance(other, LogicNodeBase):
            self.add_children(other)
        else:
            for _other in other:
                self.add_children(_other)
        return self

    def __repr__(self):
        return self.name

    def __getitem__(self, item):
        """Slow get item which find node named "item" by iterating over the all 
        nodes."""
        for node in PreOrderIter(self.root):
            if node.name == item:
                return node
        raise KeyError('Key: "{}" not in the tree.'.format(item))


class SimulationBase(CrowdDynamicsObject):
    timestamp = Instance(
        datetime,
        help='Timestamp. Defaults to time when object was first created.')
    data = Instance(
        klass=OrderedDict,
        help="Generated simulation data that is shared between logic nodes. "
             "This should be data that can be updated and should be saved on "
             "every iteration.")

    @default('data')
    def _default_data(self):
        return OrderedDict()

    @default('timestamp')
    def _default_timestamp(self):
        return datetime.now()

    @property
    def name_with_timestamp(self):
        """Name with timestamp."""
        return '_'.join(
            (self.name, self.timestamp.strftime(self._datetime_fmt)))

    def update(self):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError
