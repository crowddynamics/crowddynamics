Agents
======

Active agents
-------------
Set of active agents usually referred as just agents is denoted :math:`N = \{ 0, 1, \ldots, n-1 \}` where items are the indices of the agents. Individual agent is usually denoted :math:`i \in N` or :math:`j \in N`. There are three different model for agent displayed in the figure.

*Circular* model that is unorientable. *Elliptical* model and *three circle* model are orientable and more realistic but also more complex as they require rotational motion and more computation time to compute distances.

In this simulation model we use circular model to approximate long distance interactions and we improve short distance interactions by using three circles model.

.. image::
    ../_static/agent_model.*

Circular model
^^^^^^^^^^^^^^
Circular model models agent as *particle* that has mass :math:`m`, center of mass :math:`\mathbf{x}` and radius :math:`r`. Model is not orientable and does not require rotational motion.

Elliptical model
^^^^^^^^^^^^^^^^
In elliptical model agent is represented as a *rigid body* that has mass :math:`m`, center of mass :math:`\mathbf{x}` and two axes major :math:`r` and minor :math:`r_t`.

Parametric formula for elliptical agent

.. math::
   R(\varphi) [r_t \cos(\phi), r \sin(\phi)],

where :math:`R(\varphi)` is rotation matrix and :math:`\phi` is angle from normal vector :math:`\hat{\mathbf{e}}_n`.

Three circle model
^^^^^^^^^^^^^^^^^^
Three circle model aka multi circle model models agent as a *rigid body* that has mass :math:`m`, center of mass :math:`\mathbf{x}` and three circles which represent torso and shoulders. Torso has radius of :math:`r_t` and is centered at center of mass :math:`\mathbf{x}`. Shoulder have both radius of  :math:`r_s` and are centered at :math:`\mathbf{x} \pm r_{ts} \hat{\mathbf{e}}_t`, where :math:`\hat{\mathbf{e}}_t` is unit vector tangential to the orientation :math:`\varphi` of the body.

Three circle model is more realistic and more suitable for computing than elliptical model making it preferred choice over elliptical model. Also some approximations for ellipses are based on circular arcs. [2001fourarc]_

.. [2001fourarc] Qian, Wen-Han, and Kang Qian. "Optimising the four-arc approximation to ellipses." Computer aided geometric design 18, no. 1 (2001): 1-19.


..
   Properties
   ^^^^^^^^^^

   .. csv-table::
      :file: ../tables/body_types.csv
      :header-rows: 1

   .. csv-table::
      :file: ../tables/agent_table.csv
      :header-rows: 1


Passive agents
--------------

Areas
^^^^^

Bounds

Area inside which agents are active. If agent exits bounds they become inactive.

Goals


Obstacles
^^^^^^^^^

Linear walls

.. image::
   ../_static/wall_model.*

Linear wall is defined by two points

.. math::
   \mathbf{p}_{0}, \mathbf{p}_{1}



Round walls

Round wall is defined by point of center and radius

.. math::
   \mathbf{p}, r_{w}


Relative properties
-------------------
Relative properties between agents.

Between active agents
^^^^^^^^^^^^^^^^^^^^^

Circular

.. math::
   \tilde{\mathbf{x}} &= \mathbf{x}_{i} - \mathbf{x}_{j} \\
   \tilde{\mathbf{v}} &= \mathbf{v}_{i} - \mathbf{v}_{j} \\
   d &= \left\| \tilde{\mathbf{x}} \right\| \\
   r_{tot} &= r_i + r_j \\
   h &= d - r_{tot} \\
   \hat{\mathbf{n}} &= \tilde{\mathbf{x}} / d \\
   \hat{\mathbf{t}} &= R(-90^{\circ}) \cdot \hat{\mathbf{n}}

Three circles



Between passive and active agent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Linear wall

.. math::
   \tilde{\mathbf{x}} & \\
   \tilde{\mathbf{v}} &= \mathbf{v}_{i} \\
   \mathbf{q}_{0} &= \mathbf{x}_{i} - \mathbf{p}_{0} \\
   \mathbf{q}_{1} &= \mathbf{x}_{i} - \mathbf{p}_{1} \\
   d &= \begin{cases} \left\| \mathbf{q}_{0} \right\| & l_{t} > l_{w} \\
   \left| l_{n} \right| & \text{otherwise} \\
   \left\| \mathbf{q}_{1} \right\| & l_{t} < -l_{w}
   \end{cases} \\
   \hat{\mathbf{n}} &= \begin{cases}
   \hat{\mathbf{q}}_{0} & l_{t} > l_{w} \\
   \operatorname{sign}(l_{n})\hat{\mathbf{n}}_{w} & \text{otherwise} \\
   \hat{\mathbf{q}}_{1} & l_{t} < -l_{w}
   \end{cases} \\
   \hat{\mathbf{t}} &= R(-90^{\circ}) \cdot \hat{\mathbf{n}}

