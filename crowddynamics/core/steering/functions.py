import numba
from numba import f8

from crowddynamics.core.vector2D import normalize


@numba.jit([f8[:](f8[:], f8[:], f8)], nopython=True, nogil=True, cache=True)
def weighted_average(e0, e1, p):
    r"""Weighted average of two (unit)vectors

    .. math::
       \mathbf{\hat{e}_{out}} = 
       \mathcal{N} \big(p \mathbf{\hat{e}_{0}} + (1 - p) \mathbf{\hat{e}_{1}} \big)
    
    where
    
    - :math:`\mathcal{N}(\mathbf{v}) = \frac{\mathbf{v}}{\|\mathbf{v}\|}` is the 
      normalization of the vector

    Args:
        e0 (numpy.ndarray): Vector
        e1 (numpy.ndarray): Vector
        p (float): Weight between :math:`p \in [0, 1]`
    
    Returns:
        numpy.ndarray:
    """
    return normalize(p * e0 + (1 - p) * e1)
