import os
from functools import lru_cache

import pandas as pd


class Table:
    root = os.path.abspath(".")
    ext = ".csv"
    filenames = ("agent", "body")

    @lru_cache()
    def agent(self):
        name = "agent"
        path = os.path.join(self.root, name + self.ext)
        return pd.read_csv(path, index_col=[0])

    @lru_cache()
    def body(self):
        name = "body"
        path = os.path.join(self.root, name + self.ext)
        return pd.read_csv(path, index_col=[0])
