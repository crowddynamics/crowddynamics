Theory
======

Crowd Simulation
----------------

.. todo::
   General description about the theory and mathematics behind crowd simulations.

- Time :math:`t \in \mathbb{R}^{+}`
- Domain :math:`\Omega \subset \mathbb{R}^{2}` is surface that contains all objects in the simulation such as agents and obstacles.
- Agent :math:`\mathcal{A}`.
- Obstacle :math:`\mathcal{O}`.
- Exit / target region :math:`\mathcal{E}`.


----

.. todo::
   Documentation about different quantities related and observed in crowddynamics.


Density
-------
Crowd density :math:`\rho` is the number of agent per unit of area :math:`\mathrm{P / m^{2}}`. Realistic values range from :math:`0` to :math:`10` people per square metre. Density ranges can be classified in increasing order [gkstill]_

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

Another way to define density in unit-less quantity is to define it to be the percentage of area occupied. This has value range :math:`0` to :math:`1`.


Flow
----
Crowd flow which has similarities with granular flow, gas kinetics and fluid-dynamics measures the magnitude and how the crowd flows. Rate of flow is defined as number of agents per second

.. math::
   J = \frac{\Delta N}{\Delta t}

There are several types of crowd flow, which can be classified [Duives2014]_

.. graphviz:: crowdflow.dot

Mathematically the direction of the flow can be derived from the dot product. For agents with velocities :math:`\mathbf{v}_{i}` and :math:`\mathbf{v}_{j}` the dot product

.. math::
   \mathbf{v}_{i} \cdot \mathbf{v}_{j} = \|\mathbf{v}_{i}\| \|\mathbf{v}_{j}\| \cos(\varphi) \\

where :math:`\varphi` is the angle between the agents. In order to determine the type of the flow we must calculate the value of angle :math:`\varphi` between neighbouring agents.


Pressure
--------
Pressure :math:`p` in crowds created by contact forces :math:`\mathbf{f}_{c}` from agents pressing onto each others.

.. math::
   p \propto \sum_{i \in N_{neighbor}} \mathbf{f}_{c_i}

----

Fundamental Diagram
-------------------
Fundamental diagram offers an empirical relationship between density :math:`\rho` and velocity :math:`\mathbf{v}` of pedestrian movement. It can also be defined as the relationship between density :math:`\rho` and flow :math:`\Phi = \rho \mathbf{v}`. [Seyfried2005]_

