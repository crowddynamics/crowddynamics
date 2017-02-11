"""Communication between the GUI and simulation."""
from collections import namedtuple

import numpy as np

from crowddynamics.taskgraph import TaskNode

Config = namedtuple('Config', ['object', 'attributes'])


class GuiCommunication(TaskNode):
    """GuiCommunication"""

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation
        self.configs = []
        self.messages = []

    def set(self, configs):
        self.configs = configs
        for config in self.configs:
            self.messages.append(
                namedtuple(config.object.__name__, config.attributes)
            )

    def update(self):
        messages = []
        for config, message in zip(self.configs, self.messages):
            messages.append(
                message(*(np.copy(getattr(config.object, attr))
                          for attr in config.attributes))
            )
        self.simulation.queue.put(messages)
