Egress congestion
=================

Modeling egress and room evacuation situations through a bottleneck.

Spatial game
------------

Eight closes agents aka *Moore's neighborhood*.

Spatial game for egress congestion. Game is interaction between agents.

Estimated evacuation time for an agent

.. math::
   T_i = \frac{\lambda_i}{\beta},

where

.. math::

   \beta \propto \left \lfloor \frac{r_{door}}{r_{agent}} \right \rfloor

is the capacity of the exit door and

.. math::

   \lambda_i

number of other agents closer to the exit.

To compute :math:`\lambda = [\lambda_i]_{i\in N}` which is a vector consisting of all :math:`\lambda`'s for corresponding agents, we need to compute the distances with the center of the door

.. math::

   d_i = \| \mathbf{x}_{door} - \mathbf{x}_{i} \|

Then we sort the distances by indices to get the order of agent indices from closest to the exit door to farthest, sorting by indices again gives us number of agents closer to the exit door

.. math::

   \lambda &= \operatorname{argsort} \left(\underset{i\in N}{\operatorname{argsort}} d_i \right).


Average evacuation time

.. math::
   T_{ij} = \frac{(T_i + T_j)}{2}.

Available safe egress time

.. math::
   T_{ASET}(T) = T_{ASET}(0) - T,

where :math:`T_{ASET}(0)` is initial available safe egress time and :math:`T` is current simulation time.

Payoff matrix

+-----------+------------------------------------------+---------------+
|           |                                Impatient |       Patient |
+-----------+------------------------------------------+---------------+
| Impatient | :math:`T_{ASET}/T_{ij}, T_{ASET}/T_{ij}` | :math:`-1, 1` |
+-----------+------------------------------------------+---------------+
|   Patient |                            :math:`1, -1` | :math:`0, 0`  |
+-----------+------------------------------------------+---------------+

Updating strategies by `Poisson process`_

.. _poisson process: http://preshing.com/20111007/how-to-generate-random-timings-for-a-poisson-process/

.. literalinclude:: ../../../crowd_dynamics/core/random.py
   :pyobject: clock

----

.. [game2013] Heli??vaara, S., Ehtamo, H., Helbing, D., & Korhonen, T. (2013). Patient and impatient pedestrians in a spatial game for egress congestion. Physical Review E - Statistical, Nonlinear, and Soft Matter Physics. http://doi.org/10.1103/PhysRevE.87.012802
