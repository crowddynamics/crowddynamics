import os
from functools import lru_cache

import pandas as pd


class Load:
    root = os.path.abspath(__file__)
    root = os.path.split(root)[0]
    filenames = ("agent", "body")

    @lru_cache()
    def table(self, name):
        ext = ".csv"
        # TODO: converters. Evaluate to values.
        if name in self.filenames:
            path = os.path.join(self.root, name + ext)
            return pd.read_csv(path, index_col=[0])
