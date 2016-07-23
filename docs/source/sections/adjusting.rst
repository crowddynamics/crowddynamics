Adjusting Motion
================
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
