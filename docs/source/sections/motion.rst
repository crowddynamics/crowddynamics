Social Force Model
==================

Social force model consists of real contact forces and fictitious adjusting, social forces and random fluctuation to simulate crowd motion. [helbing1995]_, [helbing2000]_

Law's of motion are given as system of differential equations

Translational motion

.. math::
   m \frac{d^{2}}{d t^{2}} \mathbf{x}(t) = \mathbf{f}(t) + \boldsymbol{\xi}

Rotational motion

.. math::
   I \frac{d^{2}}{d t^{2}} \varphi(t) = M(t) + \eta


Total force exerted on the agent is the sum of movement adjusting, social and contact forces between other agents and wall.

.. math::
   \mathbf{f}_{i}(t) = \mathbf{f}_{i}^{adj} + \sum_{j\neq i}^{} \left(\mathbf{f}_{ij}^{soc} + \mathbf{f}_{ij}^{c}\right) + \sum_{w}^{} \left(\mathbf{f}_{iw}^{soc} + \mathbf{f}_{iw}^{c}\right)


Total torque exerted on agent, is the sum of adjusting contact and social torques

.. math::
   M_{i}(t) = M_{i}^{adj} + \sum_{j\neq i}^{} \left(M_{ij}^{soc} + M_{ij}^{c}\right) + \sum_{w}^{} \left(M_{iw}^{soc} + M_{iw}^{c}\right)


.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: motion

**References**

.. [helbing1995] Helbing, Dirk, and Peter Molnar. "Social force model for pedestrian dynamics." Physical review E 51, no. 5 (1995): 4282.

.. [helbing2000] Helbing, Dirk, Illés Farkas, and Tamas Vicsek. "Simulating dynamical features of escape panic." Nature 407, no. 6803 (2000): 487-490.












