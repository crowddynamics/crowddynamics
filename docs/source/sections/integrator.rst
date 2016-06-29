Integrator
==========

Verlet integration
------------------
System is updated using discrete time step :math:`\Delta t`

.. math::
   t_{0}, t_{1}, \ldots, t_{k} = 0, \Delta t, \ldots, t_{k-1} + \Delta t \\

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

Adaptive time step :math:`\Delta t` is used when integration is done.

.. literalinclude:: ../../../crowd_dynamics/core/integrator.py
   :pyobject: integrator
