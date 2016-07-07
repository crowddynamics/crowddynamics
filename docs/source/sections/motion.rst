Social Force Model
==================

System of differential equations
--------------------------------
Social force model uses consists of real contact forces and hypothetical adjusting, social forces and random fluctuation to simulate crowd motion.

Law's of motion are given as system of differential equations

Position and velocity

.. math::
   m \frac{d^{2}}{d t^{2}} \mathbf{x}(t) = \mathbf{f}(t)

Rotational motion

.. math::
   I \frac{d^{2}}{d t^{2}} \varphi(t) = M(t)


Total Motion
------------
Total force exerted on the agent is the sum of movement adjusting, social and contact forces between other agents and wall.

.. math::
   \mathbf{f}_{i}(t) = \mathbf{f}_{i}^{adj} + \sum_{j\neq i}^{} \left(\mathbf{f}_{ij}^{soc} + \mathbf{f}_{ij}^{c}\right) + \sum_{w}^{} \left(\mathbf{f}_{iw}^{soc} + \mathbf{f}_{iw}^{c}\right) + \boldsymbol{\xi}_{i}


Total torque exerted on agent, is the sum of adjusting contact and social torques

.. math::
   M_{i}(t) = M_{i}^{adj} + \sum_{j\neq i}^{} \left(M_{ij}^{soc} + M_{ij}^{c}\right) + \sum_{w}^{} \left(M_{iw}^{soc} + M_{iw}^{c}\right) + \eta_{i}


.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: motion

Adjusting Motion
----------------
Force adjusting agent's movement towards desired in some characteristic time

.. math::
   \mathbf{f}^{adj} = \frac{m}{\tau^{adj}} (v_{0} \cdot \hat{\mathbf{e}} - \mathbf{v})


.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_adjust

Torque adjusting agent's rotational motion towards desired

.. math::
   M_{}^{adj} = \frac{I_{rot}}{\tau_{adj}^{rot}} \left( \frac{\varphi_{}(t) - \varphi_{}^{0}}{\pi}  \omega_{}^{0} - \omega_{}(t)\right)

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: torque_adjust

Interactions
------------

Interactions between agent and another agent or wall are modeled using social and contact forces.

Agent-agent
^^^^^^^^^^^
Circular

.. math::
   \tilde{\mathbf{x}} &= \mathbf{x}_{i} - \mathbf{x}_{j} \\
   \tilde{\mathbf{v}} &= \mathbf{v}_{i} - \mathbf{v}_{j} \\
   d &= \left\| \tilde{\mathbf{x}} \right\| \\
   r_{tot} &= r_i + r_j \\
   h &= d - r_{tot} \\
   \hat{\mathbf{n}} &= \tilde{\mathbf{x}} / d \\
   \hat{\mathbf{t}} &= R(-90^{\circ}) \cdot \hat{\mathbf{n}}

Three circles



Agent-wall
^^^^^^^^^^
Linear wall

.. math::
   \tilde{\mathbf{x}} & \\
   \tilde{\mathbf{v}} &= \mathbf{v}_{i} \\
   \mathbf{q}_{0} &= \mathbf{x}_{i} - \mathbf{p}_{0} \\
   \mathbf{q}_{1} &= \mathbf{x}_{i} - \mathbf{p}_{1} \\
   d &= \begin{cases} \left\| \mathbf{q}_{0} \right\| & l_{t} > l_{w} \\
   \left| l_{n} \right| & \text{otherwise} \\
   \left\| \mathbf{q}_{1} \right\| & l_{t} < -l_{w}
   \end{cases} \\
   \hat{\mathbf{n}} &= \begin{cases}
   \hat{\mathbf{q}}_{0} & l_{t} > l_{w} \\
   \operatorname{sign}(l_{n})\hat{\mathbf{n}}_{w} & \text{otherwise} \\
   \hat{\mathbf{q}}_{1} & l_{t} < -l_{w}
   \end{cases} \\
   \hat{\mathbf{t}} &= R(-90^{\circ}) \cdot \hat{\mathbf{n}}


Social
^^^^^^
Psychological force for collision avoidance. Naive velocity independent equation

.. math::
   \mathbf{f}^{soc} = A \exp\left(-\frac{h}{B}\right) \hat{\mathbf{n}}

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_social_velocity_independent

Improved velocity dependent algorithm

.. math::
   \mathbf{f}^{soc} &= -\nabla_{\tilde{\mathbf{x}}} E(\tau) \\
   &= -\nabla_{\tilde{\mathbf{x}}} \left(\frac{k}{\tau^{2}} \exp \left( -\frac{\tau}{\tau_{0}} \right) \right) \\
   &= - \left(\frac{k}{a \tau^{2}}\right) \left(\frac{2}{\tau} + \frac{1}{\tau_{0}}\right) \exp\left (-\frac{\tau}{\tau_{0}}\right ) \left(\tilde{\mathbf{v}} -\frac{a \tilde{\mathbf{x}} + b \tilde{\mathbf{v}}}{d} \right),

where

.. math::
   a &= \tilde{\mathbf{v}} \cdot \tilde{\mathbf{v}} \\
   b &= -\tilde{\mathbf{x}} \cdot \tilde{\mathbf{v}} \\
   c &= \tilde{\mathbf{x}} \cdot \tilde{\mathbf{x}} - \tilde{r}^{2} \\
   d &= \sqrt{b^{2} - a c} \\
   \tau &= \frac{b - d}{a}.

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_social

Torque from social forces acting with other agent or wall

.. math::
   \mathbf{M}_{}^{soc} = \mathbf{r}_{}^{soc} \times \mathbf{f}_{}^{soc}


Contact
^^^^^^^
Physical contact force

.. math::
   \mathbf{f}^{c} = - h \cdot \left(\mu \cdot \hat{\mathbf{n}} - \kappa \cdot (\mathbf{v} \cdot \hat{\mathbf{t}}) \hat{\mathbf{t}}\right) + c_{n} \cdot (\mathbf{v} \cdot \hat{\mathbf{n}}) \hat{\mathbf{n}} , \quad h < 0

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_contact

Torque from contact forces acting with other agent or wall

.. math::
   \mathbf{M}_{}^{c} = \mathbf{r}_{}^{c} \times \mathbf{f}_{}^{c}


Random fluctuation
------------------
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


Integrator
----------
System is updated using discrete time step :math:`\Delta t`

.. math::
   t_{0}, t_{1}, \ldots, t_{k} = 0, \Delta t, \ldots, t_{k-1} + \Delta t \\

Adaptive time step :math:`\Delta t` is used when integration is done.

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


.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: integrator
