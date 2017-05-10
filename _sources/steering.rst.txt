Steering
========

.. list-table:: Navigation Algorithms
   :header-rows: 1

   * - Algorithm
     - Type
     - Source
   * - Shortest Path
     - Discrete
     - [Kretz2011a]_, [Cristiani2015b]_
   * - Obstacle Handling
     - Discrete
     - [Cristiani2015b]_
   * - Herding
     - Continuous
     - [Helbing2000a]_


.. automodule:: crowddynamics.core.steering.navigation
   :noindex:

.. figure:: figures/vector_field.png
   :target: _images/vector_field.png
   :alt: Vector Field

   *Example of vector field created by static potential which combines obstacle
   handling and shortest path algorithms*

.. autoclass:: crowddynamics.core.steering.navigation.static_potential
   :noindex:

.. automodule:: crowddynamics.core.steering.quickest_path
   :noindex:
   :members: distance_map, direction_map

.. automodule:: crowddynamics.core.steering.obstacle_handling
   :noindex:
   :members: obstacle_handling

.. autoclass:: crowddynamics.core.steering.functions.weighted_average
   :noindex:

.. autoclass:: crowddynamics.core.steering.herding.herding
   :noindex:


.. automodule:: crowddynamics.core.steering.orientation
   :noindex:
