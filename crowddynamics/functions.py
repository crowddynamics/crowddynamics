import importlib
import math
import os
import sys
from functools import wraps, lru_cache
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import ruamel.yaml as yaml

# TODO: yaml
numpy_options = {
    'precision': 5,
    'threshold': 6,
    'edgeitems': 3,
    'linewidth': None,
    'suppress': False,
    'nanstr': None,
    'infstr': None,
    'formatter': None
}

pandas_options = {
    'display.chop_threshold': None,
    'display.precision': 4,
    'display.max_columns': 8,
    'display.max_rows': 8,
    'display.max_info_columns': 8,
    'display.max_info_rows': 8
}

root = os.path.abspath(__file__)
root = os.path.split(root)[0]


@lru_cache()
def load_config(filename):
    configs_folder = os.path.join(root, "configs")
    name, ext = os.path.splitext(filename)
    path = os.path.join(configs_folder, filename)
    if ext == ".csv":
        return pd.read_csv(path, index_col=[0])
    elif ext == ".yaml":
        with open(path) as f:
            return yaml.load(f, Loader=yaml.Loader)
    else:
        raise Exception("Filetype not supported.")


def numpy_format():
    np.set_printoptions(**numpy_options)


def pandas_format():
    for key, val in pandas_options.items():
        pd.set_option(key, val)


def format_time(timespan, precision=3):
    """Jupyter notebook timeit time formatting.
    Formats the timespan in a human readable form"""

    if timespan >= 60.0:
        # we have more than a minute, format that in a human readable form
        # Idea from http://snipplr.com/view/5713/
        parts = [("d", 60 * 60 * 24), ("h", 60 * 60), ("min", 60), ("s", 1)]
        time = []
        leftover = timespan
        for suffix, length in parts:
            value = int(leftover / length)
            if value > 0:
                leftover %= length
                time.append(u'%s%s' % (str(value), suffix))
            if leftover < 1:
                break
        return " ".join(time)

    # Unfortunately the unicode 'micro' symbol can cause problems in
    # certain terminals.
    # See bug: https://bugs.launchpad.net/ipython/+bug/348466
    # Try to prevent crashes by being more secure than it needs to
    # E.g. eclipse is able to print a µ, but has no sys.stdout.encoding set.
    units = [u"s", u"ms", u'us', "ns"]  # the recordable value
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
        try:
            u'\xb5'.encode(sys.stdout.encoding)
            units = [u"s", u"ms", u'\xb5s', "ns"]
        except:
            pass
    scaling = [1, 1e3, 1e6, 1e9]

    if timespan > 0:
        order = min(-int(math.floor(math.log10(timespan)) // 3), 3)
    else:
        order = 3
    # return u"%.*g %s" % (precision, timespan * scaling[order], units[order])
    return u"{:.1f} {}".format(timespan * scaling[order], units[order])


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = timer()
        ret = func(*args, **kwargs)
        dt = timer() - start
        print(func.__name__, format_time(dt))
        return ret

    return wrapper
