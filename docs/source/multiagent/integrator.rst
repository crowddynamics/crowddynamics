Integrator
==========
Differential system is integrated using numerical integration scheme using discrete adaptive time step :math:`\Delta t`.

Acceleration on an agent

.. math::
   a_{k} &= \mathbf{f}_{k} / m \\
   \mathbf{x}_{k+1} &= \mathbf{x}_{k} + \mathbf{v}_{k} \Delta t + \frac{1}{2} a_{k} \Delta t^{2} \\
   \mathbf{v}_{k+1} &= \mathbf{v}_{k} + a_{k} \Delta t \\


Angular acceleration

.. math::
   \alpha_{k} &= M_{k} / I \\
   \varphi_{k+1} &= \varphi_{k} + \omega_{k} \Delta t + \frac{1}{2} \alpha_{k} \Delta t^{2} \\
   \omega_{k+1} &= \omega_{k} + \alpha_{k} \Delta t \\