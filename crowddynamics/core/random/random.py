import logging

import numpy as np
import scipy.stats

logger = logging.getLogger()
# TODO: control the random state for reproducibility


def truncnorm(start, end, loc=0.0, scale=1.0, abs_scale=None, size=1, random_state=None):
    """
    Truncated normal distribution from ``scipy.stats``.

    Args:
        start (float):
        end (float):
        loc (float):
        scale (float):
        abs_scale: Absolute scale ``scale = abs_scale / max(abs(start), abs(end))
        size (int):
        random_state (int, optional):

    Returns:
        numpy.ndarray:

    References

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    https://en.wikipedia.org/wiki/Truncated_normal_distribution
    """
    # TODO: control std
    # TODO: logger stats
    if abs_scale:
        scale = abs_scale / max(abs(start), abs(end))
    tn = scipy.stats.truncnorm.rvs(
        start, end, loc=loc, scale=scale, size=size, random_state=random_state
    )
    return tn


def random_vector(size, orient=(0.0, 2.0 * np.pi), mag=1.0):
    orientation = np.random.uniform(orient[0], orient[1], size=size)
    return mag * np.stack((np.cos(orientation), np.sin(orientation)), axis=1)
