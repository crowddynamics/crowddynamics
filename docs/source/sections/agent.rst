Agent
=====

Definition
----------
Set of agents is denoted :math:`N = \{ 0, 1, \ldots, n-1 \}` where items are the indices of the agents. Individual agent is usually denoted :math:`i \in N` or :math:`j \in N`.

Models
------
Two different model for agent can be used. Simple circular model that is unorientable and three circle model that is orientable and more realistic but also more complex.

.. image::
    ../_static/agent_model.*

Circular model
--------------
In circular model agent is represented as a *particle* that has mass :math:`m` and center of mass :math:`\mathbf{x}` and radius :math:`r`.

Three circle model
------------------
In three circle model (aka multi circle model) agent is represented as a *rigid body* that has mass :math:`m` and center of mass :math:`\mathbf{x}` and three circles which represent the torso and sholders. Torso has radius of :math:`r_t` and is centered at center of mass :math:`\mathbf{x}`. Shoulder have both radius of  :math:`r_s` and are centered at :math:`\mathbf{x} \pm r_{ts} \hat{\mathbf{e}}_t`, where :math:`\hat{\mathbf{e}}_t` is unit vector tangential to the orientation :math:`\varphi` of the body.



Properties
----------

..
   .. csv-table::
      :file: ../tables/body_types.csv
      :header-rows: 1

   .. csv-table::
      :file: ../tables/agent_table.csv
      :header-rows: 1


