Agents
======

Domain
------
Domain inside which agents are active is denoted

.. math::
   \Omega \subset \mathbb{R}^{2}


Simulation time is denoted

.. math::
   t \in \mathbb{R}^{+}

Agent
-----
Set of active agents usually referred as just agents is denoted

.. math::
   A = \{ 0, 1, \ldots, n-1 \}

where items are the indices of the agents. Individual agent is usually denoted :math:`i \in A` or :math:`j \in A`.

Size of the set

.. math::
   N = | A |

Models
^^^^^^
Three different models for she shape of the agent from above are displayed in the figure.

.. image::
    ../_static/agent_model.*

All agents have mass :math:`m > 0`, center of the mass :math:`\mathbf{x} \in \Omega` and moment of inertia :math:`I > 0`. Orientable agents have body angle of :math:`\varphi \in [-\pi, \pi]`. Normal and tangential unit vectors for agent

.. math::
   \mathbf{\hat{e}_n} &= [\cos(\varphi), \sin(\varphi)] \\
   \mathbf{\hat{e}_t} &= [\sin(\varphi), -\cos(\varphi)]

They are used in

Circular model
^^^^^^^^^^^^^^
Simplest of the models is circular model without orientation. Circle is defined with radius :math:`r > 0` from the center of mass.


Elliptical model
^^^^^^^^^^^^^^^^
Ellipse is defined two axes :math:`r` and :math:`r_t`. Elliptical model is mentioned but not used because complexity of its equation. Preferred model is three circle model which is more realistic and the fact that approximations required to compute elliptical model are based on circular arcs would make it similar to compute. [2001fourarc]_

.. [2001fourarc] Qian, W. H., & Qian, K. (2001). Optimizing the four-arc approximation to ellipses. Computer Aided Geometric Design, 18(1), 1–19. http://doi.org/10.1016/S0167-8396(00)00033-9

Three circle model
^^^^^^^^^^^^^^^^^^
Three circle model models agent with three circles which represent torso and two shoulders. Torso has radius of :math:`r_t` and is centered at center of mass :math:`\mathbf{x}` and shoulder have both radius of  :math:`r_s` and are centered at :math:`\mathbf{x} \pm r_{ts} \mathbf{\hat{e}_t}`. [langston2007]_

.. [langston2007] Langston, P. A., Masling, R., & Asmar, B. N. (2006). Crowd dynamics discrete element multi-circle model. Safety Science. http://doi.org/10.1016/j.ssci.2005.11.007


..
   Properties
   ^^^^^^^^^^

   .. csv-table::
      :file: ../tables/body_types.csv
      :header-rows: 1

   .. csv-table::
      :file: ../tables/agent_table.csv
      :header-rows: 1


Obstacle
--------
Obstacle is denoted

.. math::
   \mathcal{O} \subset \Omega

**Linear wall**

.. image::
   ../_static/wall_model.*

Linear wall is defined by two points

.. math::
   \mathbf{p}_{0}, \mathbf{p}_{1}


Exit door
---------

Exit door is denoted

.. math::
   \mathcal{E} \subset \Omega

