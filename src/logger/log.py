import logging as log
import logging.handlers
import os
import platform
import sys


def start_logging(level, filename):
    """
    Starts logger for crowd dynamics simulation.

    :param level:
    :param filename:
    :return:
    """
    # Create filename.log file
    ext = ".log"
    filename, _ = os.path.splitext(filename)
    filename += ext

    # Format of the log output
    log_format = log.Formatter('%(asctime)s, '
                               '%(levelname)s, '
                               '%(funcName)s, '
                               '%(message)s')
    logger = log.getLogger()
    logger.setLevel(level)

    file_handler = logging.handlers.RotatingFileHandler(
        filename, maxBytes=(10240 * 5), backupCount=2
    )
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    console_handler = log.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)


def user_info():
    log.info("Platform: %s", platform.platform())
    log.info("Path: %s", sys.path[0])
    log.info("Python: %s", sys.version[0:5])
