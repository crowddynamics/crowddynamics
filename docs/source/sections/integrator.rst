Integrator
==========

Explicit Euler Method
---------------------
System is updated using discrete time step

.. math::
   t_{0} = 0 \\
   t_{1} = t_{0} + \Delta t \\
   \vdots \\
   t_{k} = t_{k-1} + \Delta t


Acceleration on an agent

.. math::
   a_{k} &= \mathbf{f}_{k} / m \\
   \mathbf{v}_{k+1} &= \mathbf{v}_{k} + a_{k} \Delta t \\
   \mathbf{x}_{k+1} &= \mathbf{x}_{k} + \mathbf{v}_{k+1} \Delta t


Angular acceleration

.. math::
   \alpha_{k} &= M_{k} / I \\
   \omega_{k+1} &= \omega_{k} + \alpha_{k} \Delta t \\
   \varphi_{k+1} &= \varphi_{k} + \omega_{k+1} \Delta t
