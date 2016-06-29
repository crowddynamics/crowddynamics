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


Motion
------
Total force exerted on the agent is the sum of movement adjusting, social and contact forces between other agents and wall.

.. math::
   \mathbf{f}_{i}(t) = \mathbf{f}_{i}^{adj} + \sum_{j\neq i}^{} \left(\mathbf{f}_{ij}^{soc} + \mathbf{f}_{ij}^{c}\right) + \sum_{w}^{} \left(\mathbf{f}_{iw}^{soc} + \mathbf{f}_{iw}^{c}\right) + \boldsymbol{\xi}_{i}


Total torque exerted on agent, is the sum of adjusting contact and social torques

.. math::
   M_{i}(t) = M_{i}^{adj} + \sum_{j\neq i}^{} \left(M_{ij}^{soc} + M_{ij}^{c}\right) + \sum_{w}^{} \left(M_{iw}^{soc} + M_{iw}^{c}\right) + \eta_{i}(t)


.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: motion

Adjusting force
---------------
Force adjusting agent's movement towards desired in some characteristic time

.. math::
   \mathbf{f}^{adj} = \frac{m}{\tau^{adj}} (v_{0} \cdot \hat{\mathbf{e}} - \mathbf{v})


.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_adjust

Social force
------------
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

Contact force
-------------
Physical contact force

.. math::
   \mathbf{f}^{c} = - h \cdot \left(\mu \cdot \hat{\mathbf{n}} - \kappa \cdot (\mathbf{v} \cdot \hat{\mathbf{t}}) \hat{\mathbf{t}}\right) + c_{n} \cdot (\mathbf{v} \cdot \hat{\mathbf{n}}) \hat{\mathbf{n}} , \quad h < 0

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_contact

Fluctuation force
-----------------
From truncated normal distribution

.. math::
   \boldsymbol{\xi} &= \xi \cdot \hat{\mathbf{e}}, \quad \xi \in \mathcal{N}(\mu, \sigma^{2}), \\
   \hat{\mathbf{e}}  &= \begin{bmatrix} \cos(\varphi) & \sin(\varphi) \end{bmatrix}, \quad \varphi \in \mathcal{U}(-\pi, \pi)

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: force_random


Adjusting torque
----------------
Torque adjusting agent's rotational motion towards desired

.. math::
   M_{}^{adj} = \frac{I_{}}{\tau_{}} \left((\varphi_{}(t) - \varphi_{}^{0}) \omega_{}^{0} - \omega_{}(t)\right)

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: torque_adjust


Social torque
-------------
Torque from social forces acting with other agent or wall

.. math::
   \mathbf{M}_{}^{soc} = \mathbf{r}_{}^{soc} \times \mathbf{f}_{}^{soc}

Contact torque
--------------
Torque from contact forces acting with other agent or wall

.. math::
   \mathbf{M}_{}^{c} = \mathbf{r}_{}^{c} \times \mathbf{f}_{}^{c}

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: torque

Fluctuation torque
------------------
From truncated normal distribution

.. math::
   \eta \in \mathcal{N}(\mu, \sigma^{2})

.. literalinclude:: ../../../crowd_dynamics/core/motion.py
   :pyobject: torque_random
