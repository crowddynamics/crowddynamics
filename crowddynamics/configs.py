import os
from functools import lru_cache

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CFG_DIR = os.path.join(BASE_DIR, 'crowddynamics', 'configs')


@lru_cache()
def load_config(filename):
    name, ext = os.path.splitext(filename)
    path = os.path.join(CFG_DIR, filename)
    if ext == ".csv":
        return pd.read_csv(path, index_col=[0])
    else:
        raise Exception("Filetype not supported.")
