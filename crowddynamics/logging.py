import logging
import platform
import sys

import loggingtools

from crowddynamics.config import LOG_CFG

LOGLEVELS = [
    logging.CRITICAL, logging.FATAL, logging.ERROR, logging.WARNING,
    logging.WARN, logging.INFO, logging.DEBUG, logging.NOTSET,
    'CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'WARN', 'INFO', 'DEBUG', 'NOTSET'
]


def format_numpy(precision=5, threshold=6, edgeitems=3, linewidth=None,
                 suppress=False, nanstr=None, infstr=None, formatter=None):
    try:
        import numpy as np
        np.set_printoptions(precision, threshold, edgeitems, linewidth,
                            suppress, nanstr, infstr, formatter)
    except ImportError:
        return


pandas_options = {
    'display.chop_threshold': None,
    'display.precision': 4,
    'display.max_columns': 8,
    'display.max_rows': 8,
    'display.max_info_columns': 8,
    'display.max_info_rows': 8
}


def format_pandas(opts=pandas_options):
    try:
        import pandas as pd
        for key, val in opts.items():
            pd.set_option(key, val)
    except ImportError:
        return


def user_info():
    logger = logging.getLogger(__name__)
    logger.info("Platform: %s", platform.platform())
    logger.info("Path: %s", sys.path[0])
    logger.info("Python: %s", sys.version[0:5])


def setup_logging(loglevel=logging.INFO, log_cfg=LOG_CFG):
    loggingtools.setup_logging(log_cfg, '.logs')
    format_numpy()
    format_pandas()
    user_info()
