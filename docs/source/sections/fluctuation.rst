Random Fluctuation
==================
Fluctuation force

.. math::
   \boldsymbol{\xi} &= \xi \cdot \hat{\mathbf{e}}, \quad \xi \in \mathcal{N}(\mu, \sigma^{2}), \\
   \hat{\mathbf{e}}  &= \begin{bmatrix} \cos(\varphi) & \sin(\varphi) \end{bmatrix}, \quad \varphi \in \mathcal{U}(-\pi, \pi)

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_random

Fluctuation torque

.. math::
   \eta \in \mathcal{N}(\mu, \sigma^{2})

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: torque_random
