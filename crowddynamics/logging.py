import inspect
import math
import os
import platform
import sys
import logging.config

import functools
import timeit

import numpy as np
import pandas as pd
from ruamel import yaml

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_CFG = os.path.join(BASE_DIR, 'logging.yaml')
LOG_DIR = '.logs'

pandas_options = {
    'display.chop_threshold': None,
    'display.precision': 4,
    'display.max_columns': 8,
    'display.max_rows': 8,
    'display.max_info_columns': 8,
    'display.max_info_rows': 8
}


def format_pandas(opts=pandas_options):
    for key, val in opts.items():
        pd.set_option(key, val)


def format_numpy(precision=5, threshold=6, edgeitems=3, linewidth=None,
                 suppress=False, nanstr=None, infstr=None, formatter=None):
    np.set_printoptions(precision, threshold, edgeitems, linewidth, suppress,
                        nanstr, infstr, formatter)


def format_time(timespan, precision=3):
    """Formats the timespan in a human readable form"""

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


def setup_logging(default_path=LOG_CFG,
                  default_level=logging.INFO,
                  env_key='LOG_CFG',
                  logdir=LOG_DIR):
    """Setup logging configurations defined by dict configuration.

    References:

    .. [#] https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
    """
    path = default_path
    env = os.getenv(env_key, None)
    if env:
        path = env

    if os.path.exists(path):
        with open(path, 'rt') as file:
            config = yaml.safe_load(file.read())

        if not os.path.exists(logdir):
            os.mkdir(logdir)

        handlers_ = config['handlers']
        for name in handlers_:
            handler = handlers_[name]
            if 'filename' in handler:
                handler['filename'] = os.path.join(logdir, handler['filename'])

        # Configure logger
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

    # Nicer printing for numpy array and pandas tables
    format_numpy()
    format_pandas()


def user_info():
    logger = logging.getLogger(__name__)
    logger.info("Platform: %s", platform.platform())
    logger.info("Path: %s", sys.path[0])
    logger.info("Python: %s", sys.version[0:5])


class log_with(object):
    """Logging decorator that allows you to log with a specific logger.

    Todo:
        - loglevel
        - Pretty formatting
        - function call stack level
        - timer
        - args & kwargs
    """

    def __init__(self, logger=None, entry_msg=None, exit_msg=None):
        self.logger = logger
        self.entry_msg = entry_msg
        self.exit_msg = exit_msg

    def __call__(self, function):
        """Returns a wrapper that wraps func. The wrapper will log the entry
        and exit points of the function with logging.INFO level.
        """
        if not self.logger:
            # If logger is not set, set module's logger.
            self.logger = logging.getLogger(function.__module__)

        # Function signature
        sig = inspect.signature(function)
        arg_names = sig.parameters.keys()

        def message(args, kwargs):
            for i, name in enumerate(arg_names):
                try:
                    value = args[i]
                except IndexError:
                    # FIXME: Default values in kwargs
                    try:
                        value = kwargs[name]
                    except KeyError:
                        continue
                yield str(name) + ': ' + str(value)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            self.logger.info('<' + function.__name__ + '>' + '\n' +
                             '\n'.join(message(args, kwargs)))

            start = timeit.default_timer()
            result = function(*args, **kwargs)
            dt = timeit.default_timer() - start
            time = format_time(dt)

            self.logger.info(time)
            return result
        return wrapper
