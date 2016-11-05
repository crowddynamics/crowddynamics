Navigation
==========

- :math:`\Omega` Domain
- :math:`\mathcal{A}` Area occupied by agent
- :math:`\mathcal{O}` Area occupied by obstacle
- :math:`\mathcal{E}` Area occupied by exit
- :math:`\Omega \setminus (\mathcal{O} \cup \mathcal{A})` Unoccupied area

Distance map and static potential
---------------------------------
Navigation aka path planning is related to *continuos shortest path problem*, which can se solved by solving `Eikonal equation`_,

.. _Eikonal equation: <https://en.wikipedia.org/wiki/Eikonal_equation>

.. math::
   \left \| \nabla S(\mathbf{x}) \right \| = \frac{1}{f(\mathbf{x})}, \quad \mathbf{x} \in \Omega

where :math:`S(\mathbf{x})` is distance map, which denotes the shortest time to travel from :math:`\mathbf{x}` to destination, which is given by boundary conditions. Function :math:`f(\mathbf{x})` is the speed at :math:`\mathbf{x}` defined

.. math::
   f : \bar{\Omega} \mapsto (0, +\infty)

Boundary conditions of the distance map define the value at the the destination

.. math::
   S(\mathbf{x}) = 0, \quad \mathbf{x} \in \mathcal{E}

and inside obstacles

.. math::
   S(\mathbf{x}) \to \infty, \quad \mathbf{x} \in \mathcal{O}

We get static potential by defining speed in walkable areas :math:`\Omega \setminus \mathcal{O}` and inside obstacles :math:`\mathcal{O}`

.. math::
   f(\mathbf{x}) &= 1, \quad \mathbf{x} \in \Omega \setminus \mathcal{O} \\
   f(\mathbf{x}) &\to 0, \quad \mathbf{x} \in \mathcal{O}

Target direction

.. math::
   \hat{\mathbf{e}}_{S} = -\frac{\nabla S(\mathbf{x})}{\| \nabla S(\mathbf{x}) \|}


Dynamic potential
-----------------
Dynamics potential takes into account the positions of the agents in the field. Equation

.. math::
   \left \| \nabla T(\mathbf{x}) \right \| = \frac{1}{f(\mathbf{x})}, \quad \mathbf{x} \in \Omega

and boundary conditions

.. math::
   f(\mathbf{x}) &= 1, \quad \mathbf{x} \in \Omega \setminus (\mathcal{O} \cup \mathcal{A}) \\
   f(\mathbf{x}) &\leq 1, \quad \mathbf{x} \in \mathcal{A} \\
   f(\mathbf{x}) &\to 0, \quad \mathbf{x} \in \mathcal{O}

.. math::
   \frac{1}{f(\mathbf{x})} = 1 + \max \left( 0, c_{0} \left( 1 + c_{1} \frac{\mathbf{v} \cdot \nabla S(\mathbf{x})}{v_{0} \| \nabla S(\mathbf{x}) \|} \right) \right)

- :math:`c_{0}` general impact strength
- :math:`c_{1}` impact of the moving direction of an agent

Target direction

.. math::
   \hat{\mathbf{e}}_{T} = -\frac{\nabla T(\mathbf{x})}{\| \nabla T(\mathbf{x}) \|}


Implementation
--------------
.. Fast Marching Method.
.. Fast Iterative Method

Discretize the domain :math:`\Omega` into meshgrid.


References
----------
..
   .. bibliography:: ../bibliography/CrowdDynamics-Navigation.bib
