import numba
import numpy as np
from numba import f8, optional

from crowddynamics.core.vector2D import length_nx2


@numba.jit(f8(f8, f8, optional(f8), f8), nopython=True)
def narrow_exit_capacity(d_door, d_agent, d_layer=None, coeff=1.0):
    r"""Estimation of the capacity of narrow exit.

    Capacity estimation :math:`\beta` of unidirectional flow through narrow
    bottleneck. Capacity of the bottleneck increases in stepwise manner.

    Simple estimation

    .. math::
       \beta_{simple} = c \left \lfloor \frac{d_{door}}{d_{agent}} \right \rfloor

    More sophisticated estimation :cite:`Hoogendoorn2005a`, :cite:`Seyfried2007a`

    .. math::
       \beta_{hoogen} = c \left \lfloor \frac{d_{door} - (d_{agent} - d_{layer})}{d_{layer}} \right \rfloor

    We have relationship between simple and hoogendoorn's estimation

    .. math::
       \beta_{hoogen} = \beta_{simple}, \quad d_{layer} = d_{agent}

    Args:
        d_door (float):
            Width of the door :math:`0\,\mathrm{m} < d_{door} < 3\,\mathrm{m}`.
            Also width of the door must be larger than width of the agent
             :math:`d_{door} > d_{agent}`.

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
    assert d_door >= d_agent > 0
    if d_layer is None:
        return coeff * (d_door // d_agent)
    else:
        return coeff * ((d_door - (d_agent - d_layer)) // d_layer)


@numba.jit(nopython=True)
def agent_closer_to_exit(points, position):
    r"""
    Agent closer to exit

    Function mapping players to number of agents that are closer to the exit is denoted

    .. math::
       \lambda : P \mapsto [0, | P | - 1].

    Ordering is defined as the distances between the exit and an agent

    .. math::
       d(\mathcal{E}_i, \mathbf{x}_{i})

    where

    - :math:`\mathcal{E}_i` is the exit the agent is trying to reach
    - :math:`\mathbf{x}_{i}` is the center of the mass of an agent

    For narrow bottlenecks we can approximate the distance

    .. math::
       d(\mathcal{E}_i, \mathbf{x}_{i}) \approx \| \mathbf{c} - \mathbf{x}_{i} \|

    where

    - :math:`\| \cdot \|` is euclidean `metric`_
    - :math:`\mathbf{c}` is the center of the exit.

    .. _metric: https://en.wikipedia.org/wiki/Metric_(mathematics)

    .. Then we sort the distances by indices to get the order of agent indices from closest to the exit door to farthest, sorting by indices again gives us number of agents closer to the exit door

    Algorithm

    #) Sort by distances to map number of closer agents to player

       .. math::
           \boldsymbol{\lambda}^{-1} = \underset{i \in P}{\operatorname{arg\,sort}} \left( d(\mathcal{E}_i, \mathbf{x}_{i}) \right)

    #) Sort by players to map player to number of closer agents

    .. math::
       \boldsymbol{\lambda} = \operatorname{arg\,sort} (\boldsymbol{\lambda}^{-1})


    Args:
        points (numpy.ndarray):
            Array [[x1, y2], [x2, y2]]

        position (numpy.ndarray):
            Positions of the agents.

    Returns:
        numpy.ndarray:
            Array where indices denote agents and value how many agents are
            closer to the exit.

    """
    mid = (points[0] + points[1]) / 2.0
    dist = length_nx2(mid - position)
    # players[values] = agents, indices = number of agents closer to exit
    num = np.argsort(dist)
    # values = number of agents closer to exit, players[indices] = agents
    num = np.argsort(num)
    return num