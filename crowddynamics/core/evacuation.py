"""Evacuation related functions"""
import numba
import numpy as np
from numba.typing.typeof import typeof

from crowddynamics.core.geom2D import line_intersect
from crowddynamics.core.sensory_region import is_obstacle_between_points
from crowddynamics.core.structures import obstacle_type_linear
from crowddynamics.core.vector2D import length
from numba import i8, f8, optional

from crowddynamics.simulation.agents import NO_TARGET


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


@numba.jit((f8[:, :], f8[:, :], typeof(obstacle_type_linear)[:], f8),
           nopython=True, nogil=True, cache=True)
def exit_detection(center_door, position, obstacles, detection_range):
    """Exit detection. Detects closest exit in detection range that is in line
    of sight.

    Args:
        detection_range:
        center_door:
        position:
        obstacles:

    Returns:
        ndarray: Selected exits. Array of indices denoting which exit was
        selected. If none was selected then value is set to `not_detected = -1`.
    """
    not_detected = -1
    n = len(position)
    distance = np.full(shape=n, fill_value=detection_range, dtype=np.float64)
    detected_exit = np.full(shape=n, fill_value=not_detected, dtype=np.int64)
    """Which exit has been detected by the agent if any."""
    has_detected = np.zeros(shape=n, dtype=np.bool_)
    """False if agent has not detected an exit else True"""

    for i in range(n):
        for c in range(len(center_door)):
            # If line of sight is obstructed skip the exit
            if is_obstacle_between_points(position[i], center_door[c],
                                          obstacles):
                continue

            d = length(center_door[c] - position[i])
            if d < distance[i]:
                distance[i] = d
                detected_exit[i] = c
                has_detected[i] = True

    return detected_exit, has_detected
