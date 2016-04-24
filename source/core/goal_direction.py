import numba
import numpy as np


# @numba.jit(nopython=True, nogil=True)
def herding_behaviour(direction, herding, sight):
    """
    Update goal direction.
    """
    mean = np.mean(direction, axis=1)
    new_direction = np.zeros_like(direction)
    for i in range(len(direction)):
        new_direction[i, :] = (1 - herding) * direction[i] + herding * mean
    return new_direction
