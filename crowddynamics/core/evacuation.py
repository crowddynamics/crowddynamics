"""Evacuation related functions"""
import numba
import numpy as np
from crowddynamics.core.vector2D import length
from numba import i8, f8, optional


@numba.jit(f8(f8, f8, optional(f8), f8),
           nopython=True, nogil=True, cache=True)
def narrow_exit_capacity(d_door, d_agent, d_layer=None, coeff=1.0):
    r"""
    Capacity estimation :math:`\beta` of unidirectional flow through narrow
    bottleneck. Capacity of the bottleneck increases in stepwise manner.

    Estimation 1
        Simple estimation
        
        .. math::
           \beta_{simple} = c \left \lfloor \frac{d_{door}}{d_{agent}} \right \rfloor

    Estimation 2
        More sophisticated estimation [Hoogendoorn2005a]_, [Seyfried2007a]_

        .. math::
           \beta_{hoogen} = c \left \lfloor \frac{d_{door} - (d_{agent} - d_{layer})}{d_{layer}} \right \rfloor,\quad d_{door} \geq d_{agent}

    We have relationship between simple and Hoogendoorn's estimation

    .. math::
       \beta_{hoogen} = \beta_{simple}, \quad d_{layer} = d_{agent}

    .. list-table:: Variables
       :header-rows: 1
       
       * - Variable
         - Reference value
         - Unit
       * - :math:`d_{door} \in [0, 3]`
         - 
         - m
       * - :math:`d_{agent} > 0`
         - 
         - m
       * - :math:`d_{layer} > 0`
         - :math:`0.45`
         - m
       * - :math:`c > 0`
         -
         -

    Args:
        d_door (float):
            Width of the door :math:`d_{door}`.

        d_agent (float):
            Width of the agent :math:`d_{agent}`.

        d_layer (float, optional):
            Width of layer :math:`d_{layer}`.

        coeff (float):
            Scaling coefficient :math:`c`.

    Returns:
        float:
            Depending on the value of ``d_layer`` either simple or hoogen
            estimation is returned

            - ``None``: Uses simple estimation :math:`\beta_{simple}`
            - ``float``: Uses Hoogendoorn's estimation :math:`\beta_{hoogen}`

    """
    if d_door < d_agent:
        return 0.0
    elif d_layer is None:
        return coeff * (d_door // d_agent)
    else:
        return coeff * ((d_door - (d_agent - d_layer)) // d_layer)


@numba.jit(i8[:](f8[:], f8[:, :]),
           nopython=True, nogil=True, cache=True)
def agent_closer_to_exit(c_door, position):
    r"""Amount of positions (agents) closer to center of the exit.

    1) Denote :math:`i` as indices of the positions :math:`\mathbf{x}`.
    2) Distance from narrow exit can be estimated

      .. math::
         d_i = \| \mathbf{c} - \mathbf{x}_{i} \|

    3) By sorting the values :math:`d_i` by its indices we obtain array

      .. math::
          a_i = \underset{i \in P}{\operatorname{arg\,sort}}(d_i)

      where

      - Values: Indices of the positions sorted by the distance from the exit
      - Indices: Number of positions closer to the exit

    4) By sorting the values :math:`a_i` by it indices we obtain array

      .. math::
         \lambda_i = \operatorname{arg\,sort} (a_i)

      where

      - Values: Number of positions closer to the exit
      - Indices: Indices of the positions

    Args:
        c_door (numpy.ndarray):
            Center of the exit :math:`\mathbf{c}`.

        position (numpy.ndarray):
            Positions :math:`\mathbf{x}` of the agents.

    Returns:
        numpy.ndarray:
            Array :math:`\lambda_i`

    """
    distances = length(c_door - position)
    d_sorted = np.argsort(distances)
    num = np.argsort(d_sorted)
    return num
