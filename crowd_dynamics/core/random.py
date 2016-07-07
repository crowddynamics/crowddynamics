import numba
import numpy as np


@numba.jit(nopython=True)
def clock(interval, dt):
    """Probabilistic clock with expected frequency of update interval.
    :return: Boolean whether strategy should be updated of not.
    """
    if dt >= interval:
        return True
    else:
        return np.random.random() < (dt / interval)
