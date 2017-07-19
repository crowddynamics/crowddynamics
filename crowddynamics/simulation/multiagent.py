import logging
import multiprocessing
from multiprocessing import Process, Event

from anytree.iterators import PostOrderIter
from loggingtools import log_with
from traitlets import Instance

from crowddynamics.exceptions import CrowdDynamicsException
from crowddynamics.simulation.agents import Agents
from crowddynamics.simulation.base import SimulationBase
from crowddynamics.simulation.field import Field
from crowddynamics.simulation.logic import LogicNode


class MultiAgentSimulation(SimulationBase):
    r"""Constructing a multi-agent simulation

    Field
        Instance of :class:`Field`.

    Agents
        Instance of :class:`Agents`.

    Logic
        **Logic** of the simulation consists of tree of :class:`LogicNode`.
        Simulation is updated by calling the update function of  each logic node
        using *post-order* traversal.

    """
    field = Instance(
        Field,
        allow_none=True,
        help='')
    agents = Instance(
        Agents,
        allow_none=True,
        help='')
    logic = Instance(
        LogicNode,
        allow_none=True,
        help='')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data['iterations'] = 0
        self.data['time_tot'] = 0.0
        self.data['dt'] = 0.0

    # @log_with(timed=True, arguments=False)
    def update(self):
        """Execute new iteration cycle of the simulation."""
        for node in PostOrderIter(self.logic.root):
            node.update()
        self.data['iterations'] += 1


class MultiAgentProcess(Process):
    """Class for running MultiAgentSimulation in a new process."""
    logger = logging.getLogger(__name__)

    class EndProcess(object):
        """Marker for end of simulation"""

    def __init__(self, simulation, queue):
        """MultiAgentProcess

        Examples:
            >>> process = MultiAgentProcess(simulation, queue)
            >>> process.start()  # Starts the simulation
            >>> ...
            >>> process.stop()  # Stops the simulation

        Args:
            simulation (MultiAgentSimulation):
            queue (multiprocessing.Queue):
        """
        super(MultiAgentProcess, self).__init__()
        self.simulation = simulation
        self.exit = Event()
        self.queue = queue

    @log_with(qualname=True, ignore={'self'})
    def run(self):
        """Runs simulation process by calling update method repeatedly until
        stop is called. This method is called automatically by Process class
        when start is called."""
        self.simulation.exit_condition = lambda _: self.exit.is_set()
        try:
            self.simulation.run()
        except CrowdDynamicsException as error:
            self.logger.error(
                'Simulation stopped to error: {}'.format(error))
            self.stop()
        self.queue.put(self.EndProcess)

    @log_with(qualname=True, ignore={'self'})
    def stop(self):
        """Sets event to true in order to stop the simulation process."""
        self.exit.set()
