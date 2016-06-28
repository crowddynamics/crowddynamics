Navigation
==========

Theory
------
Navigation algorithm is a function that takes at least coordinate :math:`\mathbf{x}` as an argument and returns an unit vector :math:`\hat{\mathbf{e}}` that is used as target direction for the agent.

.. math::
   f(\mathbf{x}, \ldots) \to \hat{\mathbf{e}}.

Manual construction
-------------------


Fluid flow
----------
One way to find suitable function is to solve how *incompressible*, *irrotational* and *inviscid* fluid (ideal fluid) would flow out of the constructed space.
