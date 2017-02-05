import inspect
import os
import platform
import sys
import logging.config

import functools
import numpy as np
import pandas as pd
from ruamel import yaml


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CFG_DIR = os.path.join(BASE_DIR, 'crowddynamics', 'configs')
LOG_CFG = os.path.join(BASE_DIR, 'logging.yaml')

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


def setup_logging(default_path=LOG_CFG,
                  default_level=logging.INFO,
                  env_key='LOG_CFG',
                  logdir='.logs'):
    """Setup logging configurations. These are defined as dictConfig in
    ``default_path``.

    References:

    .. [#] https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
    """
    # Path to logging yaml configuration file.
    path = default_path

    # Set-up logging
    # logger = logging.getLogger(__name__)
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        # Load configs
        with open(path, 'rt') as file:
            config = yaml.safe_load(file.read())
        # Direct all logs into ``logdir`` directory
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
        - Set loglevel
        - Pretty formatting
    """
    def __init__(self, logger=None, loglevel=logging.INFO,
                 entry_msg=None, exit_msg=None):
        # TODO: set loglevel
        # TODO: pretty formatting
        self.logger = logger
        self.loglevel = loglevel
        self.entry_msg = entry_msg
        self.exit_msg = exit_msg

    def __call__(self, function):
        """
        Returns a wrapper that wraps func. The wrapper will log the entry
        and exit points of the function with logging.INFO level.
        """
        # set logger if it was not set earlier
        if not self.logger:
            self.logger = logging.getLogger(function.__module__)
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
            msg = function.__name__
            msg += "\n"
            msg += "\n".join(message(args, kwargs))

            self.logger.info(msg)
            if self.entry_msg:
                self.logger.info(self.entry_msg)
            f_result = function(*args, **kwargs)
            if self.exit_msg:
                self.logger.info(self.exit_msg)
            return f_result
        return wrapper
