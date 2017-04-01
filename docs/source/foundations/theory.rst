Mathematical Foundations
========================
Mathematically crowd dynamics or crowd simulation is defined as a function

.. math::
   f: A \times \mathbb{R}^+ \mapsto A

that maps current state of agent :math:`A` and time :math:`t \in \mathbb{R}^{+}` to new state :math:`A`. Common states to be manipulated are agents *spatial* and *rotational* motion. States are model dependent and more can be defined as needed, keeping in mind the principle of `Occam's razor`_.

.. _Occam's razor: https://en.wikipedia.org/wiki/Occam%27s_razor

----

Time
----
Time is denoted

.. math::
   t \in \mathbb{R}^{+}.

When numerically solving differential equation we use adaptive discrete timestep from a defined interval

.. math::
   \Delta t \in [0.1, 0.01].

In dynamics differential in respect of time is sometimes denoted

.. math::
   \frac{d^2}{dt^2} \mathbf{x} = \mathbf{\ddot{x}}.


----

Geometry
--------
Spatical geometry of the crowd simulation is defined in two dimensional space :math:`\mathbb{R}^{2}`. Geometrical constructs that are used in this simulation are *point*, *curve* and *surface*. For numerical and practical purposes simulations use simple geometric object such as polygons, linestrings and circles.

Point is defined

.. math::
   \mathbf{p} \in \mathbb{R}^{2}

Linestring is linearly connected set of points

.. math::
   \mathbf{p}_{i} \in \mathbb{R}^{2}, \quad i \in \mathbb{N}_{0}

Linerings have their end point connected as well. Polygon is a set of the area that is covered by linering. `Shapely`_.

For theoretical purposes `Bézier curves`_ may also be considered but may be too complex for simulations.

.. _Bézier curves: https://en.wikipedia.org/wiki/B%C3%A9zier_curve#General_definition

.. _Shapely: http://toblerity.org/shapely/manual.html


Domain
^^^^^^
Domain is surface that contains all objects in the simulation such as agents and obstacles. Defined

.. math::
   \Omega \subset \mathbb{R}^{2}.

Agents that move outside the domain will turn inactive.


Obstacle
^^^^^^^^
Obstacle is a static curve that is unpassable (agent cannot go through). Defined

.. math::
   \mathcal{O} \subset \Omega.


Exit
^^^^
Exit is a static curve that is passable. Defined

.. math::
   \mathcal{E} \subset \Omega.

Narrow bottleneck
^^^^^^^^^^^^^^^^^
A special case of exit is a narrow bottleneck. Defined as line from point :math:`\mathbf{p}_0` to :math:`\mathbf{p}_1`. Width of the door

.. math::
   d_{door} = \| \mathbf{p}_0 - \mathbf{p}_1 \|

Width of the exit must sufficient for agent to pass through. Lower and upper bounds for the exit width are

.. math::
   d_{agent} \leq d_{door} \leq 6 d_{agent}

Capacity estimation of unidirectional flow through narrow bottleneck. Capacity of the bottleneck increases in stepwise manner. Simple estimation of capacity

.. math::
   \beta \propto \left \lfloor \frac{d_{door}}{d_{agent}} \right \rfloor

More sophisticated estimation [hoogen2005]_, [seyfried2007]_

.. math::
   \beta \propto \left \lfloor \frac{d_{door} - (d_{agent} - d_{layer})}{d_{layer}} \right \rfloor


where

- :math:`\left \lfloor \cdot \right \rfloor` is the `floor function`_

.. _floor function: https://en.wikipedia.org/wiki/Floor_and_ceiling_functions


----


Agent
-----
Set of agents that are subject to motion usually referred just agents is denoted

.. math::
   A &= \{ a_0, a_1, \ldots \} \\
   N &= | A |

Common convention denoting individual agent is :math:`i \in A` or :math:`j \in A`. Area occupied by agents

.. math::
   \mathcal{A} = \bigcup_{i \in A} \mathcal{A}_{i} \subset \Omega.


Model
^^^^^
Three different models for she shape of the agent from above are displayed in the figure.

.. image::
    agent_model.*

All agents have mass :math:`m > 0`, center of the mass :math:`\mathbf{x} \in \Omega` and moment of inertia :math:`I > 0`. Orientable agents have body angle of :math:`\varphi \in [-\pi, \pi]`. Normal and tangential unit vectors for agent

.. math::
   \mathbf{\hat{e}_n} &= [\cos(\varphi), \sin(\varphi)] \\
   \mathbf{\hat{e}_t} &= [\sin(\varphi), -\cos(\varphi)]

Velocity of the agent's center of mass :math:`\mathbf{v}` and rotational velocity :math:`\omega` around center of mass.


Circular model
^^^^^^^^^^^^^^
Simplest of the models is circular model without orientation. Circle is defined with radius :math:`r > 0` from the center of mass.


Elliptical model
^^^^^^^^^^^^^^^^
Ellipse is defined two axes :math:`r` and :math:`r_t`. Elliptical model is mentioned but not used because complexity of its equation. Preferred model is three circle model which is more realistic and the fact that approximations required to compute elliptical model are based on circular arcs would make it similar to compute. [fourarc2001]_


Three circle model
^^^^^^^^^^^^^^^^^^
Three circle model models agent with three circles which represent torso and two shoulders. Torso has radius of :math:`r_t` and is centered at center of mass :math:`\mathbf{x}` and shoulder have both radius of  :math:`r_s` and are centered at :math:`\mathbf{x} \pm r_{ts} \mathbf{\hat{e}_t}`. [langston2007]_

----

Initial configuration
---------------------
Initial placement of agents inside a polygonal surface, a spawn denoted by :math:`\mathcal{S} \subset \Omega`, uses random uniform sampling.

#) Triangulation
#) Random Triangle
#) Random point inside triangle


----

.. [fourarc2001] Qian, W. H., & Qian, K. (2001). Optimizing the four-arc approximation to ellipses. Computer Aided Geometric Design, 18(1), 1–19. http://doi.org/10.1016/S0167-8396(00)00033-9

.. [hoogen2005] Hoogendoorn, S. P., & Daamen, W. (2005). Pedestrian Behavior at Bottlenecks. Transportation Science, 39(2), 147–159. http://doi.org/10.1287/trsc.1040.0102

.. [langston2007] Langston, P. A., Masling, R., & Asmar, B. N. (2006). Crowd dynamics discrete element multi-circle model. Safety Science. http://doi.org/10.1016/j.ssci.2005.11.007

.. [seyfried2007] Seyfried, A., Rupprecht, T., Passon, O., Steffen, B., Klingsch, W., & Boltes, M. (2007). New insights into pedestrian flow through bottlenecks. Transportation Science, 43:395–406, 43(3), 16. http://doi.org/10.1287/trsc.1090.0263

.. [obstacle2015] Cristiani, E., & Peri, D. (2015). Handling obstacles in pedestrian simulations: Models and optimization. Retrieved from http://arxiv.org/abs/1512.08528
