Adjusting
=========

Force
-----
*Adjusting* aka *driving* force accounts of agent's desire to reach a certain destination. In high crowd densities term *manoeuvring* is used.  Force affecting the agent takes form

.. math::
   \mathbf{f}_{adj} = \frac{m}{\tau_{adj}} (v_{0} \mathbf{\hat{e}_{0}} - \mathbf{v}),

where

- Characteristic time :math:`\tau_{adj}` time for agent to adjust it movement. Value :math:`0.5` is often used, but for example impatient agent that tend to push other agent more this value can be reduced.
- Target velocity :math:`v_{0}` is usually *average walking speed* for agent in its current situation.
- Target direction :math:`\mathbf{\hat{e}_{0}}` is solved by *navigation* or *path planning* algorithm. More details in the navigation section.

Torque
------

Adjusting torque account for agent's desire to rotate it orientation.

.. math::
   M_{adj} = \frac{I_{rot}}{\tau_{rot}} \left( \omega_{0} \left ( \frac{\varphi - \varphi_{0}}{\pi} \right ) - \omega\right),

where

- Angular difference :math:`\varphi - \varphi_{0}` is wrapped between interval :math:`[-\pi, \pi]` so that division by :math:`\pi` returns value between :math:`[-1, 1]`. This gives direction and magnitude for the torque.
- Characteristic time :math:`\tau_{rot}` time for agent to adjust it orientation.
- Maximum angular velocity :math:`\omega_{0}`.
- Target orientation :math:`\varphi_{0}`. In low and medium crowd densities the angle of the target direction can be sufficient for target orientation. In high crowd densities agents may twist their body differently for example to try to squeeze through narrow spaces, requiring more sophisticated algorithms.
