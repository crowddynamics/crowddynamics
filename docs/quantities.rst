Quantities
==========
This section is about different quantities related and observed in crowd dynamics.


Velocity
--------
.. todo:: About measuring velocity from crowd trajectory data


Density
-------

.. autoclass:: crowddynamics.core.quantities.density_classical
      :noindex:

.. autoclass:: crowddynamics.core.quantities.density_voronoi_1
      :noindex:

.. autoclass:: crowddynamics.core.quantities.density_voronoi_2
      :noindex:

Density ranges can be classified in increasing order [gkstill]_

.. list-table::
   :header-rows: 1

   * - Class
     - min
     - max
   * - Free flow
     - :math:`0`
     - :math:`1`
   * - Stable
     - :math:`1`
     - :math:`2.5`
   * - Capacity
     - :math:`2.5`
     - :math:`3.5`
   * - Unstable
     - :math:`3.5`
     - :math:`5.0`
   * - Congested
     - :math:`\geq 5.0`
     -


Flow
----
Crowd flow which has similarities with granular flow, gas kinetics and fluid-dynamics measures the magnitude and how the crowd flows. Rate of flow is defined as number of agents per second

.. math::
   F = \frac{\Delta N}{\Delta t}\quad \left[\frac{1}{s}\right]

There are several types of crowd flow, which can be classified [Duives2014]_

.. graphviz:: graphviz/crowdflow.dot

Mathematically the direction of the flow can be derived from the dot product. For agents with velocities :math:`\mathbf{v}_{i}` and :math:`\mathbf{v}_{j}` the dot product

.. math::
   \mathbf{v}_{i} \cdot \mathbf{v}_{j} = \|\mathbf{v}_{i}\| \|\mathbf{v}_{j}\| \cos(\varphi) \\

where :math:`\varphi` is the angle between the agents. In order to determine the type of the flow we must calculate the value of angle :math:`\varphi` between neighbouring agents.


Pressure
--------
Pressure :math:`p` in crowds created by contact forces :math:`\mathbf{f}_{c}` from agents pressing onto each others.

.. math::
   p \propto \sum_{i \in N_{neighbor}} \mathbf{f}_{c_i}


Fundamental Diagram
-------------------
Fundamental diagram offers an empirical relationship between density :math:`\rho` and velocity :math:`\mathbf{v}` of pedestrian movement. It can also be defined as the relationship between density :math:`\rho` and flow :math:`\Phi = \rho \mathbf{v}`. [Seyfried2005]_


References
----------
.. [gkstill] FIPM, Prof. 2017. "Standing Crowd Density | Prof. Dr. G. Keith Still". Gkstill.Com. Accessed February 4 2017. http://www.gkstill.com/Support/crowd-density/CrowdDensity-1.html.
.. [Seyfried2005] Seyfried, A., Steffen, B., Klingsch, W., & Boltes, M. (2005). The fundamental diagram of pedestrian movement revisited. Journal of Statistical Mechanics: Theory and Experiment, 2005(10), P10002–P10002. http://doi.org/10.1088/1742-5468/2005/10/P10002
.. [Steffen2010] Steffen, B., & Seyfried, A. (2010). Methods for measuring pedestrian density, flow, speed and direction with minimal scatter. Physica A: Statistical Mechanics and Its Applications, 389(9), 1902–1910. https://doi.org/10.1016/j.physa.2009.12.015
.. [Duives2014] Duives, D. C., Daamen, W., & Hoogendoorn, S. P. (2014). State-of-the-art crowd motion simulation models. Transportation Research Part C: Emerging Technologies. http://doi.org/10.1016/j.trc.2013.02.005
