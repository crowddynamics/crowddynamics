Quantities
==========


Density
-------
Defined

.. math::
   \rho : \Omega \times \mathbb{R}^{+} \mapsto \mathbb{R}^{+}

Number of agent per unit of area :math:`\rho(\mathbf{x}, t)` of unit :math:`\mathrm{P / m^{2}}`. Realistic values range from :math:`0` to :math:`10` people per square metre. Density ranges can be classified in increasing order

- Free flow :math:`\rho \in [0,  1]`
- Stable :math:`\rho \in [1,  2.5]`
- Capacity :math:`\rho \in [2.5,  3.5]`
- Unstable :math:`\rho \in [3.5,  5]`
- Congested :math:`\rho \geq 5`

Another way to define density in unit-less quantity is to define it to be the percentage of area occupied. This has value range :math:`0` to :math:`1`.

http://www.gkstill.com/Support/crowd-density/CrowdDensity-1.html


Flow
----
Crowd flow has similarities with granular flow, gas kinetics and fluid-dynamics.

Rate
^^^^
Number of agents per second

.. math::
   J = \frac{\Delta N}{\Delta t}

of unit :math:`\mathrm{1 / s}`.

Types
^^^^^
Uni-directional

.. math::
   \mathbf{v}_{i} \parallel \mathbf{v}_{j} \\
   \mathbf{v}_{i} \cdot \mathbf{v}_{j} \mapsto \mathbb{R}^{+}

- Bottlenecks
- Evacuation


Bi-directional

.. math::
   \mathbf{v}_{i} \parallel \mathbf{v}_{j} \\
   \mathbf{v}_{i} \cdot \mathbf{v}_{j} \mapsto \mathbb{R}^{-}

- Hallways

Orthogonal

.. math::
   \mathbf{v}_{i} \perp \mathbf{v}_{j} \\
   \mathbf{v}_{i} \cdot \mathbf{v}_{j} \mapsto 0

- Crossings


Multi-directional

.. math::
   \mathbf{v}_{i} \cdot \mathbf{v}_{j} \mapsto \mathbb{R}

- Outdoors


Pressure
--------
Crowd pressure :math:`p` created by contact forces when multiple agents press onto each others.

.. math::
   p \propto \sum \mathbf{f}_{c}

----

Fundamental Diagram
-------------------
Relationship between crowd density and flow rate.
