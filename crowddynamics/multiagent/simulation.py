"""MultiAgent Simulation"""
import logging
from copy import deepcopy
from multiprocessing import Process, Event

from crowddynamics.functions import Timed
from crowddynamics.logging import log_with
from crowddynamics.multiagent.field import Field
from crowddynamics.taskgraph import TaskNode


class MultiAgentSimulation(Field):
    """Class that calls numerical algorithms of the multi-agent simulation."""
    logger = logging.getLogger(__name__)

    def __init__(self, queue=None, name="MultiAgentSimulation"):
        """Init MultiAgentSimulation

        Args:
            queue (multiprocessing.Queue):
        """
        super(MultiAgentSimulation, self).__init__()
        self.name = name
        self.queue = queue
        self.tasks = TaskNode()
        self.iterations = int(0)

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


class MultiAgentProcess(Process):
    """MultiAgentProcess

    Class for running MultiAgentSimulation in a new process.
    """
    logger = logging.getLogger(__name__)
    # End of Simulation. Value that is injected into queue when simulation ends.
    EOS = None

    def __init__(self, simulation):
        """Init MultiAgentProcess

        Args:
            simulation (MultiAgentSimulation):
        """
        super(MultiAgentProcess, self).__init__()
        self.simulation = simulation
        self.queue = self.simulation.queue
        self.exit = Event()

    @log_with(logger, entry_msg="starting", exit_msg="ending")
    def run(self):
        """Runs simulation process by calling update method repeatedly until
        stop is called."""
        while not self.exit.is_set():
            self.simulation.update()
        self.queue.put(self.EOS)

    @log_with(logger, entry_msg="Stopping")
    def stop(self):
        """Sets event to true in order to stop the simulation process."""
        self.exit.set()
