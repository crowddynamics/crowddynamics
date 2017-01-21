import logging
from copy import deepcopy
from multiprocessing import Process, Event

import numpy as np

from crowddynamics.functions import Timed, load_config
from crowddynamics.io.hdfstore import HDFStore
from crowddynamics.multiagent.configuration import Configuration


class QueueDict:
    """Queue dict"""

    def __init__(self, producer):
        """

        Args:
            producer:
        """
        self.producer = producer
        self.dict = {}
        self.args = None

    def set(self, args):
        self.args = args

        self.dict.clear()
        for (key, key2), attrs in self.args:
            self.dict[key2] = {}
            for attr in attrs:
                self.dict[key2][attr] = None

    def fill(self, d):
        for (key, key2), attrs in self.args:
            item = getattr(self.producer, key)
            for attr in attrs:
                d[key2][attr] = np.copy(getattr(item, attr))

    def get(self):
        d = deepcopy(self.dict)
        self.fill(d)
        return d


class MultiAgentSimulation(Process, Configuration):
    """
    Class that calls numerical algorithms of the multi-agent simulation.
    """

    def __init__(self, queue=None):
        """
        MultiAgentSimulation

        Args:
            queue (multiprocessing.Queue):
        """
        super(MultiAgentSimulation, self).__init__()  # Multiprocessing
        Configuration.__init__(self)

        # Logger
        self.logger = logging.getLogger("crowddynamics.simulation")

        # Multiprocessing
        self.queue = queue
        self.exit = Event()

        # State of the simulation (types matter when saving to a file)
        self.iterations = int(0)
        self.time_tot = float(0)
        self.in_goal = int(0)
        self.dt_prev = float(0)

        # Data flow
        self.hdfstore = None  # Sends data to hdf5 file
        self.queue_items = None  # Sends data to graphics

    @property
    def name(self):
        return self.__class__.__name__

    def parameters(self):
        params = (
            "time_tot",
            "in_goal",
        )
        for p in params:
            assert hasattr(self,
                           p), "{cls} doesn't have attribute {attr}".format(
                cls=p, attr=p)
        return params

    def stop(self):
        """Sets event to true in order to stop the simulation process."""
        self.logger.info("MultiAgent Exit...")
        # self.queue.put(None)  # Poison pill. Ends simulation
        self.exit.set()

    def run(self):
        """Runs simulation process until is called. This calls the update method
        repeatedly. Finally at stop it puts poison pill (None) into the queue to
        denote last generated value."""
        self.logger.info("MultiAgent Starting")
        while not self.exit.is_set():
            self.update()

        self.queue.put(None)  # Poison pill. Ends simulation
        self.logger.info("MultiAgent Stopping")

    def configure_hdfstore(self):
        if self.hdfstore is None:
            self.logger.info("")

            # Configure hdfstore file
            self.hdfstore = HDFStore(self.name)

            # Add dataset
            parameters = load_config('parameters.yaml')

            args = self.agent, parameters['agent']
            self.hdfstore.add_dataset(*args)
            self.hdfstore.add_buffers(*args)

            args = self, parameters['simulation']
            self.hdfstore.add_dataset(*args)
            self.hdfstore.add_buffers(*args)

            self.logger.info("")
        else:
            self.logger.info("Already configured.")

    def configure_queuing(self, args):
        """

        :param args: Example [("agent", ["position", "active", "position_ls", "position_rs"])]
        :return:
        """
        # FIXME
        if self.queue is not None:
            self.logger.info("")
            self.queue_items = QueueDict(self)
            self.queue_items.set(args)
        else:
            self.logger.info("Queue is not defined.")

    @Timed("Total Simulation Time")
    def update(self):
        """Update"""
        self.agent.reset_motion()
        self.agent.reset_neighbor()
        self.task_graph.evaluate()

        # TODO: TaskNode
        # Check which agent are inside the domain
        if self.domain is not None:
            num = np.sum(self.agent.active)
            self.agent.active &= self.omega.contains_points(self.agent.position)
            num -= np.sum(self.agent.active)
            self.in_goal += num

        # Raise iteration count
        self.iterations += 1

        # Stores the simulation data into buffers and dumps buffer into file
        if self.hdfstore is not None:
            self.hdfstore.update_buffers()
            if self.iterations % 100 == 0:
                self.hdfstore.dump_buffers()

        if self.queue is not None:
            data = self.queue_items.get()
            self.queue.put(data)

    # To measure JIT compilation time of numba decorated functions.
    initial_update = Timed("Jit:")(deepcopy(update))

    try:
        # If using line_profiler decorate function.
        update = profile(update)
    except NameError:
        pass
