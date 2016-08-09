Fluctuation
===========
Fluctuation force and torque are stochastic in nature and analogous to heat in particle systems. For modeling fluctuation we use :math:`\mathcal{U}(a, b)` for `uniform distribution`_ and :math:`\mathcal{N}(\mu, \sigma^{2})` for truncated `normal distribution`_.


.. _uniform distribution: https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)

.. _normal distribution: https://en.wikipedia.org/wiki/Normal_distribution


Force
-----
Fluctuation force

.. math::
   \boldsymbol{\xi} = \xi \cdot \mathbf{\hat{e}}(\mu), \quad \xi \in \mathcal{N}(\mu, \sigma^{2}), \quad \varphi \in \mathcal{U}(-\pi, \pi)

where :math:`\mathbf{\hat{e}}(\mu)` is unit vector to direction of :math:`\mu`.

.. literalinclude:: ../../../src/core/motion.py
   :pyobject: force_fluctuation


Torque
------
Fluctuation torque

.. math::
   \eta \in \mathcal{N}(\mu, \sigma^{2})

.. literalinclude:: ../../../src/core/motion.py
   :pyobject: torque_fluctuation
