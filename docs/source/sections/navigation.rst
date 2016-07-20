Navigation
==========

Target angle and direction
--------------------------
Navigation modifies agents target angle :math:`\varphi_{0}` and target direction :math:`\hat{\mathbf{e}}_{0}`.

Direction update is a function that takes at agent class as an argument and returns an unit vector :math:`\hat{\mathbf{e}}`

.. math::
   \operatorname{direction\_update}(\mathrm{agent}) \to \hat{\mathbf{e}}

and angle update returns angle

.. math::
   \operatorname{angle\_update}(\mathrm{agent}) \to [-\pi, \pi].


Navigator
---------

Navigator takes as argument functions to update target angle and direction and updates then when called.

.. literalinclude:: ../../../crowd_dynamics/core/navigation.py
   :pyobject: navigator

Default update algorithm is updating target angle to angle of target direction

.. literalinclude:: ../../../crowd_dynamics/core/navigation.py
   :pyobject: direction_to_target_angle

Update function
---------------

Manually
^^^^^^^^



Fluid flow
^^^^^^^^^^
One way to find suitable function is to solve how *incompressible*, *irrotational* and *inviscid* fluid (ideal fluid) would flow out of the constructed space.


