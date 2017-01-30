import logging
from copy import deepcopy
from multiprocessing import Process, Event

from crowddynamics.functions import Timed
from crowddynamics.logging import log_with
from crowddynamics.multiagent.field import Field
from crowddynamics.taskgraph import TaskNode


class MultiAgentSimulation(Process, Field):
    r"""
    Class that calls numerical algorithms of the multi-agent simulation.
    """
    logger = logging.getLogger(__name__)
    # End of Simulation. Value that is injected into queue when simulation ends.
    EOS = None

    def __init__(self, queue=None):
        """
        MultiAgentSimulation

        Args:
            queue (multiprocessing.Queue):
        """
        super(MultiAgentSimulation, self).__init__()
        Field.__init__(self)

        # Multiprocessing
        self.queue = queue
        self.exit = Event()

        # Algorithms
        # TODO: potential subclassing
        self.tasks = TaskNode()

        # State of the simulation (types matter when saving to a file)
        self.iterations = int(0)

    @property
    def name(self):
        return self.__class__.__name__

    @log_with(logger, entry_msg="starting", exit_msg="ending")
    def run(self):
        """Runs simulation process by calling update method repeatedly until
        stop is called."""
        while not self.exit.is_set():
            self.update()
        self.queue.put(self.EOS)

    @log_with(logger, entry_msg="Stopping")
    def stop(self):
        """Sets event to true in order to stop the simulation process."""
        self.exit.set()

    @Timed("Total Simulation Time")
    def update(self):
        """Execute new iteration cycle of the simulation."""
        self.tasks.evaluate()
        self.iterations += 1

    # To measure JIT compilation time of numba decorated functions.
    initial_update = Timed("jit:")(deepcopy(update))

    # If using line_profiler decorate function.
    try:
        update = profile(update)
    except NameError:
        pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
