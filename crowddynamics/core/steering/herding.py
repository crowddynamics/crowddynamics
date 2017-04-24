"""Herding / Flocking / Leader-Follower effect"""
import numba
import numpy as np
from numba import f8

from crowddynamics.core.vector.vector2D import normalize


@numba.jit(f8[:](f8[:], f8[:, :], f8), nopython=True, nogil=True, cache=True)
def herding(e0, e0_neigh, p):
    r"""Herding effect
    
    .. math::
       \mathbf{\hat{e}_{herding}} = 
           \mathcal{N} \big((1 - p_i) \mathbf{\hat{e}_{0}}_i + 
           p_i \left\langle\mathbf{\hat{e}_{0}}_j\right\rangle_{i} \big)

    where 
    
    - :math:`\langle \mathbf{\hat{e}_{0}}_j \rangle` is the average of 
      directions of neighbours of agent :math:`i`. 
    - :math:`\mathcal{N}(\mathbf{v}) = \frac{\mathbf{v}}{\|\mathbf{v}\|}` is the 
      normalization of the vector

    Args:
        e0 (numpy.ndarray):
            Agents desired direction
            
        e0_neigh (numpy.ndarray):
            Directions of the neighbouring agents

        p (float): 
            Degree of herding :math:`p_i \in [0, 1]`. Indicates how much herding 
            behaviour agent experiences

            - :math:`p_i = 0`: No herding
            - :math:`p_i = 1`: Total herding

    Returns:
        numpy.ndarray: 
            New direction vector :math:`\mathbf{\hat{e}_{herding}}`
    """
    n = e0_neigh.shape[1]
    s = np.zeros(shape=n)
    for i in range(n):
        s += e0_neigh[i, :]
    return normalize((1 - p) * e0 + p * s / n)
