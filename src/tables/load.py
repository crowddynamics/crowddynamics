import os
from functools import lru_cache

import pandas as pd


class Table:
    root = os.path.abspath(__file__)
    root = os.path.split(root)[0]
    ext = ".csv"
    filenames = ("agent", "body")

    @lru_cache()
    def load(self, name):
        # TODO: converters. Evaluate to values.
        if name in self.filenames:
            path = os.path.join(self.root, name + self.ext)
            return pd.read_csv(path, index_col=[0])
