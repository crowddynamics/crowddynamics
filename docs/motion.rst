Motion
======

.. list-table:: Motion Algorithms
   :header-rows: 1

   * - Algorithm
     - Type
     - Source
   * - Integrator
     -
     -
   * - Collision Avoidance / Social Force
     -
     - [Karamouzas2014b]_
   * - Contact Force and Torque
     -
     - [Helbing2000a]_, [Langston2006]_, [Korhonen2008b]_
   * - Adjusting Force and Torque
     -
     - [Helbing2000a]_, [Korhonen2008b]_
   * - Fluctuation and Torque
     -
     - [Helbing2000a]_, [Korhonen2008b]_


.. automodule:: crowddynamics.core.integrator
   :noindex:

.. automodule:: crowddynamics.core.motion.power_law
   :noindex:
   :members: magnitude, gradient_circle_circle, gradient_three_circle,
             force_social_circular, force_social_three_circle

.. automodule:: crowddynamics.core.motion.contact
   :noindex:

.. autoclass:: crowddynamics.core.motion.contact.force_contact
   :noindex:

.. automodule:: crowddynamics.core.motion.fluctuation
   :noindex:

.. autoclass:: crowddynamics.core.motion.fluctuation.force_fluctuation
   :noindex:

.. autoclass:: crowddynamics.core.motion.fluctuation.torque_fluctuation
   :noindex:

.. automodule:: crowddynamics.core.motion.adjusting
   :noindex:

.. autoclass:: crowddynamics.core.motion.adjusting.force_adjust
   :noindex:

.. autoclass:: crowddynamics.core.motion.adjusting.torque_adjust
   :noindex:
