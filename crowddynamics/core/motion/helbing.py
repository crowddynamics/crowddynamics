import numba
import numpy as np
from numba import f8


@numba.jit([f8[:](f8, f8[:], f8, f8)], nopython=True, nogil=True, cache=True)
def force_social_helbing(h, n, a, b):
    r"""Helbing's model's original social force. Independent of the velocity or
    direction of the agent. [Helbing2000a]_

    .. math::
       A \exp\left(-\frac{h}{B}\right) \mathbf{\hat{n}}

    Args:
        h (float):
            Skin-to-skin distance between agents

        n (numpy.ndarray):
            Normal unit vector

        a (float):
            Constant :math:`A = 2 \cdot 10^{3} \,\mathrm{N}`

        b (float):
            Constant :math:`B = 0.08 \,\mathrm{m}`

    Returns:
        numpy.ndarray: Social force vector
    """
    return a * np.exp(- h / b) * n
