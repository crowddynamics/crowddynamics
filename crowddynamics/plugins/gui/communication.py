from copy import deepcopy

import numpy as np


class QueueDict:
    """Queue dict"""
    # TODO: Communication module

    def __init__(self, producer):
        """
        Init queue dict

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
