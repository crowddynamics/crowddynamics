import importlib
import math
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from functools import wraps, lru_cache
from timeit import default_timer as timer

import pandas as pd
from ruamel import yaml

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CFG_DIR = os.path.join(BASE_DIR, 'crowddynamics', 'configs')


@lru_cache()
def load_config(filename):
    name, ext = os.path.splitext(filename)
    path = os.path.join(CFG_DIR, filename)
    if ext == ".csv":
        return pd.read_csv(path, index_col=[0])
    elif ext == ".yaml":
        with open(path) as f:
            return yaml.load(f, Loader=yaml.Loader)
    else:
        raise Exception("Filetype not supported.")


def create_parameters():
    from crowddynamics.multiagent.agent import spec_agent

    ext = ".yaml"
    name = "parameters"
    filepath = os.path.join(CFG_DIR, name + ext)

    # TODO: Add Comments
    default = OrderedDict([("resizable", False),
                           ("graphics", False)])

    data = OrderedDict([('simulation', OrderedDict()),
                        ('agent', OrderedDict()), ])

    for item in spec_agent:
        data['agent'][item[0]] = deepcopy(default)

    # Mutable values that are stored
    resizable = ("position", "angle", "active")
    for item in resizable:
        data['agent'][item]['resizable'] = True

    # Values to be updated in graphics
    graphics = ("position", "angle", "active")
    for item in graphics:
        data['agent'][item]['graphics'] = True

    parameters = ("time_tot", "in_goal")
    for item in parameters:
        data['simulation'][item] = deepcopy(default)

    resizable = ("time_tot", "in_goal")
    for item in resizable:
        data['simulation'][item]['resizable'] = True

    with open(filepath, "w") as file:
        yaml.dump(data,
                  stream=file,
                  Dumper=yaml.RoundTripDumper,
                  default_flow_style=False)


class Timed:
    """
    Decorator for timing function execution time.
    """

    def __init__(self, msg=""):
        """

        Args:
            msg (str): Message to be printed when function is executed.
        """
        self.msg = msg

    def __call__(self, f):
        msg = self.msg

        @wraps(f)
        def wrapper(*args, **kwargs):
            start = timer()
            ret = f(*args, **kwargs)
            dt = timer() - start

            print(msg, format_time(dt))

            return ret

        return wrapper


def public(f):
    """Use a decorator to avoid retyping function/class names.

    * Based on an idea by Duncan Booth:
    http://groups.google.com/group/comp.lang.python/msg/11cbb03e09611b8a
    * Improved via a suggestion by Dave Angel:
    http://groups.google.com/group/comp.lang.python/msg/3d400fb22d8a42e1

    See StackOverflow post:

    https://stackoverflow.com/questions/6206089/is-it-a-good-practice-to-add-names-to-all-using-a-decorator

    Args:
        f (object): Object to be set

    Returns:
        object:
    """
    all = sys.modules[f.__module__].__dict__.setdefault('__all__', [])
    if f.__name__ not in all:  # Prevent duplicates if run from an IDE.
        all.append(f.__name__)
    return f


def run_simulation(name, iterations):
    """
    Run simulation

    Args:
        name (str):
        iterations (int):
    """
    configs = load_config("simulations.yaml")
    simu_dict = configs["simulations"][name]
    module_name = simu_dict["module"]
    class_name = simu_dict["class"]
    kwargs = simu_dict["kwargs"]

    module = importlib.import_module(module_name)
    simulation = getattr(module, class_name)
    process = simulation(None, **kwargs)

    process.initial_update()
    for _ in range(iterations):
        process.update()


def format_time(timespan, precision=3):
    """ Jupyter notebook timeit time formatting.
    Formats the timespan in a human readable form
    """

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