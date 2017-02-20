"""Evacuation related functions"""
import numba
import numpy as np
from numba import i8, f8, optional

from crowddynamics.core.vector import length_nx2


@numba.jit(f8(f8, f8, optional(f8), f8), nopython=True)
def narrow_exit_capacity(d_door, d_agent, d_layer=None, coeff=1.0):
    r"""Estimation of the capacity of narrow exit.

    Capacity estimation :math:`\beta` of unidirectional flow through narrow
    bottleneck. Capacity of the bottleneck increases in stepwise manner.

    Simple estimation

    .. math::
       \beta_{simple} = c \left \lfloor \frac{d_{door}}{d_{agent}} \right \rfloor

    More sophisticated estimation [Hoogendoorn2005a]_, [Seyfried2007a]_
    when :math:`d_{door} \geq d_{agent}`

    .. math::
       \beta_{hoogen} = c \left \lfloor \frac{d_{door} - (d_{agent} - d_{layer})}{d_{layer}} \right \rfloor

    We have relationship between simple and hoogendoorn's estimation

    .. math::
       \beta_{hoogen} = \beta_{simple}, \quad d_{layer} = d_{agent}

    Args:
        d_door (float):
            Width of the door :math:`0\,\mathrm{m} < d_{door} < 3\,\mathrm{m}`.

        d_agent (float):
            Width of the agent :math:`d_{agent} > 0\,\mathrm{m}`.

        d_layer (float, optional):
            Width of layer :math:`d_{layer}`. Reference value:
            :math:`0.45\,\mathrm{m}`

        coeff (float):
            Scaling coefficient :math:`c > 0` for the estimation.

    Returns:
        float:
            Depending on the value of ``d_layer`` either simple or hoogen
            estimation is returned

            - ``None``: Uses simple estimation :math:`\beta_{simple}`
            - ``float``: Uses Hoogendoorn's estimation :math:`\beta_{hoogen}`

    """
    if d_door < d_agent:
        return 0.0

    if d_layer is None:
        return coeff * (d_door // d_agent)
    else:
        return coeff * ((d_door - (d_agent - d_layer)) // d_layer)


@numba.jit(i8[:](f8[:], f8[:, :]), nopython=True)
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
    distances = length_nx2(c_door - position)
    d_sorted = np.argsort(distances)
    num = np.argsort(d_sorted)
    return num
