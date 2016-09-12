Game Theoretical Models
=======================

Spatial Game for Egress Congestion
----------------------------------
Spatial `game`_ for egress congestion between players :math:`P \subset A`

.. math::
   (S, f)

where

- :math:`S` is the set of strategies
- :math:`f : S \times S \mapsto \mathbb{R}` is the payoff function

Individual strategy is denoted

.. math::
   s_{i} \in S, \quad i \in P

.. _game: https://en.wikipedia.org/wiki/Normal-form_game

Payoff Matrix
-------------
+-----------+------------------------------------------+---------------+
|           |                                Impatient |       Patient |
+-----------+------------------------------------------+---------------+
| Impatient | :math:`T_{aset}/T_{ij}, T_{aset}/T_{ij}` | :math:`-1, 1` |
+-----------+------------------------------------------+---------------+
|   Patient |                            :math:`1, -1` | :math:`0, 0`  |
+-----------+------------------------------------------+---------------+

Set of strategies

.. math::
   S &= \{ \text{Impatient}, \text{Patient} \} \\
     &= \{ 0, 1 \}

Payoff function

.. math::
   f(s_i, s_j) =
   \begin{cases}
   \begin{cases}
   T_{aset}/T_{ij} \\
   1
   \end{cases} \\
   \begin{cases}
   -1 \\
   0
   \end{cases}
   \end{cases}


Available Safe Egress Time

.. math::
   T_{aset}(t) = T_{0} - t,

where

- :math:`T_{0}` is initial available safe egress time
- :math:`t` is current simulation time.

Estimated evacuation time for an agent

.. math::
   T_i = \frac{\lambda_i}{\beta},

where

- :math:`\beta` is the capacity of the exit door
- :math:`\lambda_i` number of other agents closer to the exit.

Average evacuation time

.. math::
   T_{ij} = \frac{(T_i + T_j)}{2}.


Strategy Selection
------------------
New strategy is selected using best response dynamics

.. math::
   s_{i} = \underset{s \in S}{\arg \min} \sum_{j \in N_{i}^{neigh}} f(s, s_{j})

where

- :math:`N_{i}^{neigh} \subset P \setminus \{P_{i}\}` set is eight closes agents at maximum skin-to-skin distance of :math:`0.40 \ \mathrm{m}` from agent :math:`i`
- :math:`\arg \min` function `argmin`_

.. _argmin: https://en.wikipedia.org/wiki/Arg_max


Strategy Updating
-----------------
Updating strategies by `Poisson process`_

.. _poisson process: http://preshing.com/20111007/how-to-generate-random-timings-for-a-poisson-process/

Agents Closer to Exit
---------------------
.. To compute :math:`\lambda = [\lambda_i]_{i\in N}` which is a vector consisting of all :math:`\lambda`'s for corresponding agents, we need to compute the

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


Effect on agents
----------------
- :math:`k`
- :math:`\tau_{adj}`
- :math:`\sigma_{force}`

----

.. [game2013] Heli??vaara, S., Ehtamo, H., Helbing, D., & Korhonen, T. (2013). Patient and impatient pedestrians in a spatial game for egress congestion. Physical Review E - Statistical, Nonlinear, and Soft Matter Physics. http://doi.org/10.1103/PhysRevE.87.012802


.. [game2014] Von Schantz, A., & Ehtamo, H. (2014). Cellular automaton evacuation model coupled with a spatial game. In Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics). http://doi.org/10.1007/978-3-319-09912-5_31