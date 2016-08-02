Adjusting Motion
================

*Adjusting* aka *driving* force accounts of agent's desire to reach a certain destination. In high crowd densities term *manoeuvring* is used.  Force affecting the agent takes form

.. math::
   \mathbf{f}_{adj} = \frac{m}{\tau_{adj}} (v_{0} \cdot \mathbf{\hat{e}_{0}} - \mathbf{v}),

where

- Characteristic time :math:`\tau_{adj}` time for agent to adjust it movement. Value :math:`0.5` is often used, but for example impatient agent that tend to push other agent more this value can be reduced.
- Target velocity :math:`v_{0}` is usually *average walking speed* for agent in its current situation.
- Target direction :math:`\mathbf{\hat{e}_{0}}` is solved by *navigation* or *path planning* algorithm. More details in the navigation section.

----

Adjusting torque account for agent's desire to rotate it orientation.

.. math::
   M_{adj} = \frac{I_{rot}}{\tau_{rot}} \left( \omega_{0} \left ( \frac{\varphi - \varphi_{0}}{\pi} \right ) - \omega\right),

where

- Angular difference :math:`\varphi - \varphi_{0}` is wrapped between interval :math:`[-\pi, \pi]` so that division by :math:`\pi` returns value between :math:`[-1, 1]`. This gives direction and magnitude for the torque.
- Characteristic time :math:`\tau_{rot}` time for agent to adjust it orientation.
- Maximum angular velocity :math:`\omega_{0}`.
- Target orientation :math:`\varphi_{0}`. In low and medium crowd densities the angle of the target direction can be sufficient for target orientation. In high crowd densities agents may twist their body differently for example to try to squeeze through narrow spaces, requiring more sophisticated algorithms.

..
   .. literalinclude:: ../../../crowd_dynamics/core/motion.py
      :pyobject: force_adjust


   .. literalinclude:: ../../../crowd_dynamics/core/motion.py
      :pyobject: torque_adjust

----

Navigation
----------

Distance map
^^^^^^^^^^^^

Navigation aka path planning is related to *continuos shortest path problem*, which can se solved by solving `Eikonal equation`_,

.. math::
   \left \| \nabla S(\mathbf{x}) \right \| = \frac{1}{f(\mathbf{x})}, \quad \mathbf{x} \in \Omega

where :math:`S(\mathbf{x})` is distance map, which denotes the shortest time to travel from :math:`\mathbf{x}` to destination, which is given by boundary conditions. Function :math:`f(\mathbf{x})` is the speed at :math:`\mathbf{x}` defined

.. math::
   f : \bar{\Omega} &\mapsto (0, +\infty)

Boundary conditions of the distance map define the value at the the destination

.. math::
   S(\mathbf{x}) &= 0, \quad \mathbf{x} \in \mathcal{E}

and inside obstacles

.. math::
   S(\mathbf{x}) &\to \infty, \quad \mathbf{x} \in \mathcal{O}

Static potential
^^^^^^^^^^^^^^^^
We get static potential by defining speed in walkable areas :math:`\Omega \setminus \mathcal{O}` and inside obstacles :math:`\mathcal{O}`

.. math::
   f(\mathbf{x}) &= 1, \quad \mathbf{x} \in \Omega \setminus \mathcal{O} \\
   f(\mathbf{x}) &\to 0, \quad \mathbf{x} \in \mathcal{O}

Dynamic potential
^^^^^^^^^^^^^^^^^

.. math::
   f(\mathbf{x}) &= 1, \quad \mathbf{x} \in \Omega \setminus (\mathcal{O} \cup \mathcal{A}) \\
   f(\mathbf{x}) &\leq 1, \quad \mathbf{x} \in \mathcal{A} \\
   f(\mathbf{x}) &\to 0, \quad \mathbf{x} \in \mathcal{O}

.. math::
   \frac{1}{f(\mathbf{x})} &= 1 + \max \left( 0, c_{0} \left( 1 + c_{1} \frac{\mathbf{v} \cdot \nabla S(\mathbf{x})}{v_{0} \| \nabla S(\mathbf{x}) \|} \right) \right)

Target direction
^^^^^^^^^^^^^^^^

.. math::
   \hat{\mathbf{e}}_{0} &= -\frac{\nabla S(\mathbf{x})}{\| \nabla S(\mathbf{x}) \|}


.. _Eikonal equation: <https://en.wikipedia.org/wiki/Eikonal_equation>

----

Orientation
-----------
Target orientation :math:`\varphi_{0}`


.. [quickpath2011] Kretz, T., Große, A., Hengst, S., Kautzsch, L., Pohlmann, A., & Vortisch, P. (2011). Quickest Paths in Simulations of Pedestrians. Advances in Complex Systems, 14(5), 733–759. http://doi.org/10.1142/S0219525911003281

.. [dense2016] Stüvel, S. A. (2016). Dense Crowds of Virtual Humans.
