Neighborhoods
=============

Neighborhood of agent :math:`i` is denoted :math:`N_i^{neigh} \subset N \setminus \{i\}` is a subset of agents whose distance from agent :math:`i` is smaller than :math:`d \geq 0`

.. math::
   N_i^{neigh} = \{j \mid j \in N, j \neq i,  \quad \operatorname{dist}(i, j) \leq d  \}

Distance function can be defined in different ways depending on the use. For example, distance from the center of masses

.. math::
   \operatorname{dist}(i, j) &= \| x_i - x_j \| \\
                             &= \| \tilde{x} \|

or skin-to-skin distance of agents

.. math::
   \operatorname{dist}(i, j) = h_{ij}.

We can also limit the size of the neighborhood :math:`|N_{neigh}|` to contain certain number of the closest agents.

Maximum limit of the neighbourhood size is a packing problem. We use following equation to estimate the limit

.. math::
   | N_{neigh} | \leq \rho \frac{A_{neigh}}{A_{agent}}

where :math:`\rho` is packing density coefficient denoting of the maximum percentage of area that can be filled by the agents.


Neighborhood can used for speeding up iterating over agents for updating *social forces*, for *egress congestion* algorithms or *herding* algorithms.

Number of iterations over all agents with naive algorithm has computational complexity of

.. math::
   \frac{(|N| - 1)^2}{2} \in \mathcal{O}(|N|^2)

Using neighborhood we have

.. math::
   |N| | N_{neigh} | \in \mathcal{O}(|N|)


When :math:`| N | > | N_{neigh} |` we can obtain speed up. We have to keep in mind that naive updation of the neighborhood is still :math:`\mathcal{O}(|N|^2)`. Updation of the neighbourhood can be splitted to be partially updated every interval to avoid large jumps in computational time for smoother visualization for example.


Computing neighborhood
----------------------
Algorithm for computing neighborhood.

* Define shape of the neighborhood
* Define distance algorithm
* Define distance :math:`d`
* Define size limit


Updating neighborhood
---------------------
Neighborhood needs to be updated at some time interval :math:`t \geq \Delta t` that is larger than the integration timestep. Higher value reduces computational cost but increases error.

Maximum agent velocity :math:`v_{max}`

- Maximum error of social force :math:`\varepsilon`
- Higher speeds require larger neighborhoods (at least in the direction of the velocity).


Circular neighborhood
---------------------
For circular agents in circular neighborhood

.. math::
   |N_{neigh}| \leq \rho \left(\frac{R}{r}\right)^2

where

* :math:`R` is the radius of the neighborhood
* :math:`r` is the mean radius of the agents
* Estimates for :math:`\rho \in [0.7, 0.8]` can be obtained from `circle packing in a circle`_ problem.

.. _circle packing in a circle: https://en.wikipedia.org/wiki/Circle_packing_in_a_circle

Social force error without scaling parameter :math:`k` is the distance of the magnitude from zero

.. math::
   \varepsilon &= \| \mathbf{f}^{soc} / k \| - 0 \\
               &= \left(\frac{1}{a \tau^{2}}\right) \left(\frac{2}{\tau} + \frac{1}{\tau_{0}}\right) \exp\left (-\frac{\tau}{\tau_{0}}\right ) \left\| \left(\tilde{\mathbf{v}} -\frac{a \tilde{\mathbf{x}} + b \tilde{\mathbf{v}}}{d} \right) \right\| \\

Maximum error :math:`\max(\varepsilon)` is found when two agents are in head to head collision, mathematically

.. math::
   \varphi = \angle(\tilde{\mathbf{x}}, \tilde{\mathbf{v}}) = \pi

Head on collision minimizes time-to-collision

.. math::
   \tau = \tau_{min} = \frac{\| \tilde{\mathbf{x}} \| - \tilde{r}}{\| \tilde{\mathbf{v}} \|}

Vector part of the eqation

.. math::
   \max\left(\left\| \left(\tilde{\mathbf{v}} -\frac{a \tilde{\mathbf{x}} + b \tilde{\mathbf{v}}}{d} \right) \right\| \right)  = \| \tilde{\mathbf{v}} \|

Error

.. math::
   \varepsilon = \frac{1}{\| \tilde{\mathbf{v}} \|} \left(\frac{1}{\tau_{min}^{2}}\right) \left(\frac{2}{\tau_{min}} + \frac{1}{\tau_{0}}\right) \exp\left (-\frac{\tau_{min}}{\tau_{0}}\right )

By subsituting

.. math::
   \| \tilde{\mathbf{x}} \| &= R - 2 t v_{max} \\
   \| \tilde{\mathbf{v}} \| &= 2 v_{max} \\
   \tilde{r} &= 2 r

.. math::
   \tau_{min} = \frac{R/2 - r}{v_{max}} - t

.. math::
   r &\in [0.21, 0.27] \\
   v_{max} &\leq 1.5 \\
   t &\geq \Delta t

.. math::
   \varepsilon(R, v_{max}, t) = \frac{1}{2 v_{max}} \left(\frac{1}{\tau_{min}^{2}}\right) \left(\frac{2}{\tau_{min}} + \frac{1}{\tau_{0}}\right) \exp\left (-\frac{\tau_{min}}{\tau_{0}}\right )
