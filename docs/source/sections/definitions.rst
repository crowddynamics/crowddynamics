Definitions
===========

Time
----
Time is denoted

.. math::
   t \in \mathbb{R}^{+}.

When numerically solving differential equation we use adaptive discrete timestep

.. math::
   \Delta t \in [0.1, 0.01].

In dynamics differential in respect of time is sometimes denoted

.. math::
   \frac{d^2}{dt^2} \mathbf{x} &= \mathbf{\ddot{x}}.


----


Domain
------
Domain containing agent and obstacles.

.. math::
   \bar{\Omega} \subset \mathbb{R}^{2}

Domain can be broken down to open area and boundary

.. math::
   \bar{\Omega} &= \Omega \cup \partial\Omega

----


Agent
-----
Set of active agents that are subject to motion usually referred just agents is denoted

.. math::
   A = \{ 0, 1, \ldots, n-1 \}

where items are the indices of the agents. Individual agent is usually denoted :math:`i \in A` or :math:`j \in A`.

Size of the set

.. math::
   N = | A |

Area occupied by agents

.. math::
   \mathcal{A} &= \sum_{i \in A} \mathcal{A}_{i}, \quad \mathcal{A}_{i} \subset \Omega

----

Models
^^^^^^
Three different models for she shape of the agent from above are displayed in the figure.

.. image::
    ../_static/agent_model.*

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




..
   Properties
   ^^^^^^^^^^

   .. csv-table::
      :file: ../tables/body_types.csv
      :header-rows: 1

   .. csv-table::
      :file: ../tables/agent_table.csv
      :header-rows: 1


----

Obstacle
--------
Obstacle is denoted

.. math::
   \mathcal{O} \subset \Omega

..
    .. image::
       ../_static/wall_model.*


Linear curves
^^^^^^^^^^^^^
Piecewise linear curve or more formally `polygonal chain`_ that consists of linearly connected points

.. _polygonal chain: https://en.wikipedia.org/wiki/Polygonal_chain

.. math::
   \mathbf{p}_{i} \in \mathbb{R}^{2}, \quad i \in \{0, \ldots, n\}.



Bezier curves
^^^^^^^^^^^^^
`Bézier curve`_ are parametric curves

.. _Bézier curve: https://en.wikipedia.org/wiki/B%C3%A9zier_curve#General_definition

.. math::
   \mathbf {B} (t)={}&\sum _{i=0}^{n}{n \choose i}(1-t)^{n-i}t^{i}\mathbf {p} _{i}



----

Exit
----

Exit is denoted

.. math::
   \mathcal{E} \subset \Omega


----

.. [fourarc2001] Qian, W. H., & Qian, K. (2001). Optimizing the four-arc approximation to ellipses. Computer Aided Geometric Design, 18(1), 1–19. http://doi.org/10.1016/S0167-8396(00)00033-9

.. [langston2007] Langston, P. A., Masling, R., & Asmar, B. N. (2006). Crowd dynamics discrete element multi-circle model. Safety Science. http://doi.org/10.1016/j.ssci.2005.11.007

.. [obstacle2015] Cristiani, E., & Peri, D. (2015). Handling obstacles in pedestrian simulations: Models and optimization. Retrieved from http://arxiv.org/abs/1512.08528
