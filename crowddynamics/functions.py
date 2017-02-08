import os
import sys
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


def public(f):
    """Use a decorator to avoid retyping function/class names. [#]_

    * Based on an idea by Duncan Booth:
      http://groups.google.com/group/comp.lang.python/msg/11cbb03e09611b8a

    * Improved via a suggestion by Dave Angel:
      http://groups.google.com/group/comp.lang.python/msg/3d400fb22d8a42e1

    References:

    .. [#] https://stackoverflow.com/questions/6206089/is-it-a-good-practice-to-add-names-to-all-using-a-decorator

    Args:
        f (object): Object to be set

    Returns:
        object:
    """
    all = sys.modules[f.__module__].__dict__.setdefault('__all__', [])
    if f.__name__ not in all:  # Prevent duplicates if run from an IDE.
        all.append(f.__name__)
    return f


def line_profiler(func):
    """Line profiler decorator

    - Line profiler (kernprof)
    - Memory profiler

    Decorates a function with ``profile`` if it is in the namespace
    https://github.com/rkern/line_profiler
    """
    try:
        return profile(func)
    except NameError:
        return func
