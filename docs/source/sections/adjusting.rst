Adjusting Motion
================

Force
-----

Force adjusting agent's movement towards desired in some characteristic time

.. math::
   \mathbf{f}^{adj} = \frac{m}{\tau^{adj}} (v_{0} \cdot \hat{\mathbf{e}} - \mathbf{v})


.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_adjust

Torque adjusting agent's rotational motion towards desired

.. math::
   M_{}^{adj} = \frac{I_{rot}}{\tau_{adj}^{rot}} \left( \frac{\varphi_{}(t) - \varphi_{}^{0}}{\pi}  \omega_{}^{0} - \omega_{}(t)\right)

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: torque_adjust


Navigation
----------

Target angle and direction
^^^^^^^^^^^^^^^^^^^^^^^^^^

Navigation modifies agents target angle :math:`\varphi_{0}` and target direction :math:`\hat{\mathbf{e}}_{0}`.

Direction update is a function that takes at agent class as an argument and returns an unit vector :math:`\hat{\mathbf{e}}`

.. math::
   f_{\hat{\mathbf{e}}}(\mathrm{agent}) \to \hat{\mathbf{e}}

and angle update returns angle

.. math::
   f_{\varphi}(\mathrm{agent}) \to [-\pi, \pi].


Navigator
^^^^^^^^^

Navigator takes as argument functions to update target angle and direction and updates then when called.

.. literalinclude:: ../../../crowd_dynamics/core/navigation.py
   :pyobject: navigator

Default update algorithm is updating target angle to angle of target direction

.. literalinclude:: ../../../crowd_dynamics/core/navigation.py
   :pyobject: direction_to_target_angle


Algorithm
^^^^^^^^^

.. [quickpath2011] Kretz, T., Große, A., Hengst, S., Kautzsch, L., Pohlmann, A., & Vortisch, P. (2011). Quickest Paths in Simulations of Pedestrians. Advances in Complex Systems, 14(5), 733–759. http://doi.org/10.1142/S0219525911003281
